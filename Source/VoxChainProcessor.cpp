// VoxChainProcessor.cpp — Нативный VST3 с аппаратным интерфейсом для Ableton Live (JUCE 8)
// Активный пресет при экспорте: Travis Scott  Utopia / Rodeo Chain
// Zero Latency Mode: ENABLED (0 samples)
// Цепочка из веб-рэка:
//  1. TUNE (ON)
//  2. NOISE RED (ON)
//  3. PRO-Q (ON)
//  4. DE-ESSER (ON)
//  5. FET COMP (ON)
//  6. CL 1B (ON)
//  7. TAPE (ON)
//  8. 201 ECHO (ON)
//  9. GHOST REV (ON)
//  10. LIMITER (ON)

#include <cmath>
#include <atomic>
#include <array>
#include <vector>
#include <algorithm>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <juce_gui_basics/juce_gui_basics.h>

// ============================================================================
// 1. REAL-TIME PITCH SHIFTER / AUTO-TUNE ENGINE (ZERO-ALLOCATION)
// ============================================================================
class VoxPitchShifter {
public:
    void prepare(double sr) {
        sampleRate = sr > 0.0 ? sr : 44100.0;
        std::fill(ring.begin(), ring.end(), 0.0f);
        std::fill(detectBuf.begin(), detectBuf.end(), 0.0f);
        writeIdx = 0;
        detectIdx = 0;
        phase = 0.0f;
        curRatio = 1.0f;
        tgtRatio = 1.0f;
    }

    float processSample(float x, bool enabled, int key, int scaleIdx, float retuneMs, float mix, bool zeroLat,
                        std::atomic<int>& outMidi, std::atomic<int>& outCents, std::atomic<float>& outFreq) {
        ring[writeIdx] = x;

        detectBuf[detectIdx++] = x;
        if (detectIdx >= (int)detectBuf.size()) {
            detectIdx = 0;
            if (enabled) {
                float f = detectPitch();
                if (f > 75.0f && f < 880.0f) {
                    float mf = 69.0f + 12.0f * std::log2(f / 440.0f);
                    int tm = nearestScaleMidi(mf, key, scaleIdx);
                    float tf = 440.0f * std::pow(2.0f, (tm - 69.0f) / 12.0f);
                    tgtRatio = std::clamp(tf / f, 0.5f, 2.0f);
                    outMidi.store(tm, std::memory_order_relaxed);
                    outCents.store((int)std::round((mf - (float)tm) * 100.0f), std::memory_order_relaxed);
                    outFreq.store(f, std::memory_order_relaxed);
                } else {
                    tgtRatio = 1.0f;
                    outFreq.store(0.0f, std::memory_order_relaxed);
                    outMidi.store(-1, std::memory_order_relaxed);
                }
            }
        }

        float y = x;
        if (enabled && mix > 0.001f) {
            const float win = zeroLat ? 360.0f : 920.0f;
            const float smooth = retuneMs <= 1.0f ? 0.25f : std::min(0.20f, 1.0f / (retuneMs * 0.65f + 4.0f));
            curRatio += (tgtRatio - curRatio) * smooth;
            float rate = (1.0f - curRatio) / win;
            phase = std::fmod(phase + rate + 10.0f, 1.0f);
            float p2 = std::fmod(phase + 0.5f, 1.0f);
            float w1 = 0.5f - 0.5f * std::cos(6.2831853f * phase);
            float w2 = 0.5f - 0.5f * std::cos(6.2831853f * p2);
            float shifted = readLin(phase * win + 4.0f) * w1 + readLin(p2 * win + 4.0f) * w2;
            y = x * (1.0f - mix) + shifted * mix;
        }

        writeIdx = (writeIdx + 1) & (kBufSize - 1);
        return y;
    }

private:
    static constexpr int kBufSize = 8192;
    double sampleRate = 44100.0;
    std::array<float, kBufSize> ring{};
    std::array<float, 1024> detectBuf{};
    int writeIdx = 0;
    int detectIdx = 0;
    float phase = 0.0f;
    float curRatio = 1.0f;
    float tgtRatio = 1.0f;

    float readLin(float delay) const {
        float p = (float)writeIdx - delay;
        while (p < 0.0f) p += (float)kBufSize;
        int i0 = ((int)p) & (kBufSize - 1);
        int i1 = (i0 + 1) & (kBufSize - 1);
        float frac = p - std::floor(p);
        return ring[i0] * (1.0f - frac) + ring[i1] * frac;
    }

    float detectPitch() const {
        float rms = 0.0f;
        for (float v : detectBuf) rms += v * v;
        if (std::sqrt(rms / (float)detectBuf.size()) < 0.005f) return 0.0f;
        int minLag = std::max(20, (int)(sampleRate / 850.0));
        int maxLag = std::min((int)(sampleRate / 75.0), (int)detectBuf.size() - 360);
        int bestLag = -1;
        float bestCorr = 0.0f;
        for (int lag = minLag; lag < maxLag; ++lag) {
            float c = 0.0f, n1 = 0.0f, n2 = 0.0f;
            for (int i = 0; i < 340; i += 2) {
                float a = detectBuf[i], b = detectBuf[i + lag];
                c += a * b; n1 += a * a; n2 += b * b;
            }
            float val = c / (std::sqrt(n1 * n2) + 1e-8f);
            if (val > bestCorr) { bestCorr = val; bestLag = lag; }
        }
        return (bestCorr > 0.52f && bestLag > 0) ? (float)(sampleRate / (double)bestLag) : 0.0f;
    }

    static bool inScale(int pc, int scaleIdx) {
        static const int minorS[] = {0,2,3,5,7,8,10};
        static const int majorS[] = {0,2,4,5,7,9,11};
        static const int pentaS[] = {0,3,5,7,10};
        if (scaleIdx == 2) return true; // Chromatic
        if (scaleIdx == 1) { for (int s : majorS) if (s == pc) return true; return false; }
        if (scaleIdx == 3) { for (int s : pentaS) if (s == pc) return true; return false; }
        for (int s : minorS) if (s == pc) return true;
        return false;
    }

    static int nearestScaleMidi(float mf, int key, int scaleIdx) {
        int base = (int)std::round(mf);
        int best = base;
        float bestDist = 999.0f;
        for (int off = -6; off <= 6; ++off) {
            int cand = base + off;
            int pc = ((cand - key) % 12 + 12) % 12;
            if (inScale(pc, scaleIdx)) {
                float d = std::abs((float)cand - mf);
                if (d < bestDist) { bestDist = d; best = cand; }
            }
        }
        return best;
    }
};

// ============================================================================
// 2. ARTIST PRESETS DATA STRUCTURE (16 WESTERN & RUSSIAN CHAINS)
// ============================================================================
struct ArtistPreset {
    const char* name;
    bool zeroLat;
    bool tuneOn; int tuneKey; int tuneScale; float tuneRetune; float tuneMix;
    bool gateOn; float gateThr; float gateRed;
    bool eqOn; float hpf; float mud; float pres; float high;
    bool deEssOn; float deEssFreq; float deEssAmt;
    bool compOn; int compType; float compThr; float compRatio; float compMakeup;
    bool colorOn; float tapeDrive; float airGain; float clipDrive; float colorMix;
    bool spaceOn; float dblMix; float dlyTime; float dlyFb; float dlyMix;
    bool revOn; int revType; float revSize; float revMix; float limCeil;
};

static const std::array<ArtistPreset, 16> kArtistPresets = {{
    { "01  Travis Scott - Utopia / Rodeo",  true,  true, 9,0, 3.0f, 100.0f,  true,-45.0f,65.0f,  true,105.0f,-3.5f,3.5f,4.5f,  true,6800.0f,55.0f,  true,0,-24.0f,8.0f,4.5f,   true,35.0f,4.5f,2.0f,60.0f,  true,25.0f,340.0f,44.0f,28.0f,  true,1,0.78f,30.0f,-0.5f },
    { "02  The Weeknd - After Hours Silk",  false, true, 0,0,24.0f,  78.0f,  true,-50.0f,45.0f,  true, 95.0f,-2.0f,2.5f,4.0f,  true,6600.0f,45.0f,  true,1,-20.0f,3.5f,4.0f,   true,28.0f,6.5f,1.0f,65.0f,  true,35.0f,320.0f,36.0f,24.0f,  true,0,0.82f,36.0f,-0.5f },
    { "03  Drake - OVO Warm & Intimate",    true,  true, 2,0,32.0f,  65.0f,  true,-48.0f,50.0f,  true, 85.0f,-1.0f,2.0f,2.5f,  true,6400.0f,50.0f,  true,2,-20.0f,3.5f,4.0f,   true,30.0f,3.0f,1.5f,55.0f,  true,12.0f,270.0f,25.0f,18.0f,  true,0,0.55f,20.0f,-0.5f },
    { "04  Playboi Carti - Opium / Rage",   true,  true, 5,0, 0.0f, 100.0f,  true,-42.0f,70.0f,  true,135.0f,-4.5f,5.0f,6.0f,  true,7200.0f,55.0f,  true,3,-26.0f,12.0f,5.0f,  true,52.0f,5.5f,6.5f,70.0f,  true,40.0f,240.0f,40.0f,22.0f,  true,1,0.68f,24.0f,-0.3f },
    { "05  Yeat - Twizzy Bell & Sub",       true,  true, 7,0, 2.0f, 100.0f,  true,-44.0f,65.0f,  true,120.0f,-3.5f,4.5f,5.0f,  true,7000.0f,50.0f,  true,0,-26.0f,10.0f,5.5f,  true,42.0f,5.0f,5.5f,65.0f,  true,38.0f,290.0f,38.0f,22.0f,  true,1,0.74f,26.0f,-0.4f },
    { "06  Billie Eilish - Whisper Close",  false, false,9,0,40.0f,   0.0f,  true,-52.0f,55.0f,  true, 75.0f, 1.0f,2.0f,3.5f,  true,6500.0f,40.0f,  true,1,-24.0f,3.2f,5.0f,   true,22.0f,7.0f,0.5f,50.0f,  true,45.0f,220.0f,18.0f,12.0f,  true,1,0.65f,24.0f,-0.5f },
    { "07  Post Malone - Melodic Vibrato",  false, true, 9,1,12.0f,  92.0f,  true,-48.0f,55.0f,  true,100.0f,-2.5f,3.5f,4.0f,  true,6800.0f,48.0f,  true,2,-20.0f,4.5f,4.5f,   true,32.0f,5.0f,2.0f,55.0f,  true,30.0f,310.0f,34.0f,22.0f,  true,0,0.76f,34.0f,-0.5f },
    { "08  Kendrick Lamar - Crisp Punch",   true,  false,9,0,35.0f,   0.0f,  true,-46.0f,75.0f,  true,100.0f,-2.5f,4.0f,3.0f,  true,7200.0f,60.0f,  true,4,-22.0f,6.0f,4.5f,   true,24.0f,3.5f,2.5f,45.0f,  true, 8.0f,250.0f,20.0f,10.0f,  true,0,0.38f,12.0f,-0.5f },
    { "09  Miyagi & Endshpil - Deep Warm",  false, true, 9,0,26.0f,  75.0f,  true,-48.0f,50.0f,  true, 80.0f,-1.5f,2.5f,3.0f,  true,6500.0f,45.0f,  true,1,-22.0f,3.8f,4.0f,   true,36.0f,3.5f,1.5f,62.0f,  true,22.0f,360.0f,42.0f,26.0f,  true,1,0.76f,30.0f,-0.5f },
    { "10  Big Baby Tape - Trap Punch",     true,  true, 0,0,14.0f,  82.0f,  true,-44.0f,65.0f,  true,115.0f,-3.5f,4.5f,5.0f,  true,7000.0f,58.0f,  true,3,-24.0f,10.0f,4.5f,  true,40.0f,4.5f,6.0f,60.0f,  true,20.0f,280.0f,30.0f,18.0f,  true,0,0.52f,18.0f,-0.4f },
    { "11  Skriptonit - Lo-Fi Analog Grit", false, false,9,0,30.0f,   0.0f,  true,-46.0f,50.0f,  true, 95.0f, 1.5f,3.0f,-1.0f, true,6200.0f,40.0f,  true,3,-24.0f,8.0f,4.0f,   true,58.0f,1.5f,4.0f,78.0f,  true,28.0f,310.0f,38.0f,28.0f,  true,1,0.70f,28.0f,-0.6f },
    { "12  MORGENSHTERN - Crisp Aggressive",true,  true, 2,0, 8.0f,  94.0f,  true,-42.0f,70.0f,  true,125.0f,-4.0f,5.0f,6.0f,  true,7100.0f,55.0f,  true,0,-25.0f,12.0f,5.5f,  true,42.0f,6.5f,6.5f,65.0f,  true,36.0f,260.0f,30.0f,20.0f,  true,0,0.48f,18.0f,-0.3f },
    { "13  Oxxxymiron - Dry Rap Clarity",   true,  false,9,0,40.0f,   0.0f,  true,-44.0f,80.0f,  true,105.0f,-3.0f,4.5f,3.0f,  true,6800.0f,65.0f,  true,0,-24.0f,8.0f,5.0f,   true,20.0f,3.0f,2.0f,40.0f,  true, 0.0f,240.0f,15.0f, 5.0f,  true,0,0.28f, 8.0f,-0.5f },
    { "14  Kizaru - Dark Memphis Trap",     true,  true, 5,0,16.0f,  80.0f,  true,-45.0f,60.0f,  true,100.0f,-2.5f,3.5f,3.5f,  true,6600.0f,50.0f,  true,2,-22.0f,5.0f,4.5f,   true,38.0f,3.5f,3.0f,55.0f,  true,18.0f,300.0f,36.0f,22.0f,  true,1,0.68f,22.0f,-0.5f },
    { "15  Tri Dnya Dozhdya - Rock Grit",   false, true, 9,0,20.0f,  78.0f,  true,-46.0f,60.0f,  true,110.0f,-3.0f,4.0f,4.5f,  true,6900.0f,52.0f,  true,3,-24.0f,10.0f,4.5f,  true,48.0f,4.5f,4.5f,65.0f,  true,38.0f,290.0f,34.0f,22.0f,  true,0,0.72f,32.0f,-0.4f },
    { "16  Zero-Latency Live Recording",    true,  true, 9,0,22.0f,  75.0f,  true,-48.0f,60.0f,  true, 90.0f,-2.0f,2.5f,3.0f,  true,6800.0f,45.0f,  true,0,-18.0f,4.0f,3.0f,   true,18.0f,3.0f,1.0f,40.0f,  true,12.0f,280.0f,22.0f,12.0f,  true,0,0.45f,16.0f,-0.5f }
}};

// ============================================================================
// 3. MAIN AUDIO PROCESSOR (ALL MODULES + ZERO LATENCY)
// ============================================================================
class VoxChainProProcessor : public juce::AudioProcessor
{
public:
    VoxChainProProcessor()
        : AudioProcessor(BusesProperties()
            .withInput("Input", juce::AudioChannelSet::stereo(), true)
            .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
          apvts(*this, nullptr, "PARAMS", createLayout()) {}

    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout() {
        std::vector<std::unique_ptr<juce::RangedAudioParameter>> p;
        auto addB = [&](const char* id, const char* nm, bool def) {
            p.push_back(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{id, 1}, nm, def));
        };
        auto addF = [&](const char* id, const char* nm, float mn, float mx, float def) {
            p.push_back(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{id, 1}, nm, mn, mx, def));
        };
        auto addC = [&](const char* id, const char* nm, juce::StringArray choices, int def) {
            p.push_back(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{id, 1}, nm, choices, def));
        };

        // Master
        addB("zeroLatency", "Zero Latency", true);
        addB("globalBypass", "Bypass Rack", false);
        addF("inGain", "Input dB", -18.0f, 18.0f, 0.0f);
        addF("outGain", "Output dB", -18.0f, 18.0f, 0.0f);
        addF("mix", "Global Mix", 0.0f, 100.0f, 100.0f);

        // 1. TUNE
        addB("tuneOn", "Tune On", true);
        addC("tuneKey", "Tune Key", {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"}, 9);
        addC("tuneScale", "Tune Scale", {"Minor","Major","Chromatic","Pentatonic"}, 0);
        addF("tuneRetune", "Retune ms", 0.0f, 100.0f, 12.0f);
        addF("tuneMix", "Tune Mix", 0.0f, 100.0f, 90.0f);

        // 2. NOISE RED
        addB("gateOn", "Noise Red On", true);
        addF("gateThr", "Gate Thr", -60.0f, -15.0f, -45.0f);
        addF("gateRed", "Reduction", 0.0f, 100.0f, 65.0f);

        // 3. PRO-Q
        addB("eqOn", "Pro-Q On", true);
        addF("hpf", "Low Cut Hz", 20.0f, 260.0f, 95.0f);
        addF("mud", "Mud 300Hz", -12.0f, 12.0f, -2.5f);
        addF("pres", "Pres 3.2k", -12.0f, 12.0f, 3.0f);
        addF("high", "High 9k", -12.0f, 12.0f, 3.0f);

        // 4. DE-ESSER
        addB("deEssOn", "De-Esser On", true);
        addF("deEssFreq", "De-Ess Freq", 4500.0f, 9500.0f, 6800.0f);
        addF("deEssAmt", "Suppress", 0.0f, 100.0f, 50.0f);

        // 5. DYNAMICS (FET / LA-2A / CL 1B / DISTRESS / SSL)
        addB("compOn", "Comp On", true);
        addC("compType", "Comp Model", {"FET 1176","LA-2A","CL 1B","DISTRESS","SSL STRIP"}, 0);
        addF("compThr", "Threshold", -40.0f, 0.0f, -22.0f);
        addF("compRatio", "Ratio", 2.0f, 20.0f, 6.0f);
        addF("compMakeup", "Makeup dB", 0.0f, 14.0f, 4.0f);

        // 6. ANALOG COLOR (TAPE / HEAT / AIR / CLIP)
        addB("colorOn", "Color On", true);
        addF("tapeDrive", "Tape/Heat", 0.0f, 100.0f, 32.0f);
        addF("airGain", "Air 12k dB", 0.0f, 12.0f, 4.5f);
        addF("clipDrive", "Clip dB", 0.0f, 12.0f, 2.0f);
        addF("colorMix", "Color Mix", 0.0f, 100.0f, 60.0f);

        // 7. SPACE (DOUBLER / CE-1 / 201 ECHO / TAPE DELAY)
        addB("spaceOn", "Space On", true);
        addF("dblMix", "Doubler/CE1", 0.0f, 100.0f, 25.0f);
        addF("dlyTime", "201 Time ms", 80.0f, 650.0f, 310.0f);
        addF("dlyFb", "Feedback", 0.0f, 75.0f, 36.0f);
        addF("dlyMix", "Echo Mix", 0.0f, 100.0f, 24.0f);

        // 8. REVERB (LUMEN / GHOST) + LIMITER
        addB("revOn", "Reverb On", true);
        addC("revType", "Reverb Model", {"LUMEN BRIGHT","GHOST DARK"}, 0);
        addF("revSize", "Decay Size", 0.15f, 0.95f, 0.68f);
        addF("revMix", "Reverb Mix", 0.0f, 100.0f, 26.0f);
        addF("limCeil", "Lim Ceil dB", -6.0f, 0.0f, -0.5f);

        return { p.begin(), p.end() };
    }

    void applyArtistPreset(int idx) {
        if (idx < 0 || idx >= (int)kArtistPresets.size()) return;
        currentPresetIdx = idx;
        const auto& pr = kArtistPresets[(size_t)idx];
        auto setP = [&](const char* id, float val) {
            if (auto* param = apvts.getParameter(id))
                param->setValueNotifyingHost(param->convertTo0to1(val));
        };
        setP("zeroLatency", pr.zeroLat ? 1.0f : 0.0f);
        setP("tuneOn", pr.tuneOn ? 1.0f : 0.0f);
        setP("tuneKey", (float)pr.tuneKey);
        setP("tuneScale", (float)pr.tuneScale);
        setP("tuneRetune", pr.tuneRetune);
        setP("tuneMix", pr.tuneMix);
        setP("gateOn", pr.gateOn ? 1.0f : 0.0f);
        setP("gateThr", pr.gateThr);
        setP("gateRed", pr.gateRed);
        setP("eqOn", pr.eqOn ? 1.0f : 0.0f);
        setP("hpf", pr.hpf);
        setP("mud", pr.mud);
        setP("pres", pr.pres);
        setP("high", pr.high);
        setP("deEssOn", pr.deEssOn ? 1.0f : 0.0f);
        setP("deEssFreq", pr.deEssFreq);
        setP("deEssAmt", pr.deEssAmt);
        setP("compOn", pr.compOn ? 1.0f : 0.0f);
        setP("compType", (float)pr.compType);
        setP("compThr", pr.compThr);
        setP("compRatio", pr.compRatio);
        setP("compMakeup", pr.compMakeup);
        setP("colorOn", pr.colorOn ? 1.0f : 0.0f);
        setP("tapeDrive", pr.tapeDrive);
        setP("airGain", pr.airGain);
        setP("clipDrive", pr.clipDrive);
        setP("colorMix", pr.colorMix);
        setP("spaceOn", pr.spaceOn ? 1.0f : 0.0f);
        setP("dblMix", pr.dblMix);
        setP("dlyTime", pr.dlyTime);
        setP("dlyFb", pr.dlyFb);
        setP("dlyMix", pr.dlyMix);
        setP("revOn", pr.revOn ? 1.0f : 0.0f);
        setP("revType", (float)pr.revType);
        setP("revSize", pr.revSize);
        setP("revMix", pr.revMix);
        setP("limCeil", pr.limCeil);
    }

    void prepareToPlay(double sr, int blockSize) override {
        sampleRate = sr > 0.0 ? sr : 44100.0;
        juce::dsp::ProcessSpec spec { sampleRate, (juce::uint32)std::max(1, blockSize), 2 };
        hpf.prepare(spec);
        mudFlt.prepare(spec);
        presFlt.prepare(spec);
        highFlt.prepare(spec);
        airFlt.prepare(spec);
        deEssFlt.prepare(spec);
        comp.prepare(spec);
        reverb.prepare(spec);
        pitchShifter.prepare(sampleRate);

        int maxDly = (int)(sampleRate * 1.2) + 64;
        dlyBufL.assign((size_t)maxDly, 0.0f);
        dlyBufR.assign((size_t)maxDly, 0.0f);
        dlyWrite = 0;
        dlyLpL = 0.0f; dlyLpR = 0.0f;
        gateEnv = 1.0f;
        chorusPhase = 0.0f;
        updateLatency();
    }

    void updateLatency() {
        bool zl = apvts.getRawParameterValue("zeroLatency")->load() > 0.5f;
        setLatencySamples(zl ? 0 : 256);
    }

    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override {
        juce::ScopedNoDenormals noDenormals;
        updateLatency();

        const int numCh = buffer.getNumChannels();
        const int numSamples = buffer.getNumSamples();
        if (numCh == 0 || numSamples == 0) return;

        auto val = [&](const char* id) { return apvts.getRawParameterValue(id)->load(); };

        const float inLinear = juce::Decibels::decibelsToGain(val("inGain"));
        const float outLinear = juce::Decibels::decibelsToGain(val("outGain"));
        const bool bypassed = val("globalBypass") > 0.5f;
        const bool zeroLat = val("zeroLatency") > 0.5f;
        const float globalMix = bypassed ? 0.0f : (val("mix") / 100.0f);

        buffer.applyGain(inLinear);

        float peakIn = 0.0f;
        for (int ch = 0; ch < numCh; ++ch)
            peakIn = std::max(peakIn, buffer.getMagnitude(ch, 0, numSamples));
        meterInPeak.store(peakIn, std::memory_order_relaxed);

        if (bypassed) {
            buffer.applyGain(outLinear);
            meterOutPeak.store(peakIn * outLinear, std::memory_order_relaxed);
            return;
        }

        juce::AudioBuffer<float> dryBuf;
        dryBuf.makeCopyOf(buffer);

        // 1. AUTO-TUNE (TUNE)
        const bool tuneOn = val("tuneOn") > 0.5f;
        if (tuneOn) {
            int key = (int)val("tuneKey");
            int scl = (int)val("tuneScale");
            float ret = val("tuneRetune");
            float tmix = val("tuneMix") / 100.0f;
            auto* l = buffer.getWritePointer(0);
            auto* r = numCh > 1 ? buffer.getWritePointer(1) : nullptr;
            for (int i = 0; i < numSamples; ++i) {
                float mono = r ? 0.5f * (l[i] + r[i]) : l[i];
                float tuned = pitchShifter.processSample(mono, true, key, scl, ret, tmix, zeroLat,
                                                         meterMidi, meterCents, meterFreq);
                l[i] = tuned;
                if (r) r[i] = tuned;
            }
        } else {
            meterFreq.store(0.0f, std::memory_order_relaxed);
            meterMidi.store(-1, std::memory_order_relaxed);
        }

        // 2. NOISE REDUCTION (GATE)
        if (val("gateOn") > 0.5f) {
            float thrLin = juce::Decibels::decibelsToGain(val("gateThr"));
            float minG = std::max(0.03f, 1.0f - val("gateRed") / 100.0f);
            bool openNow = false;
            for (int i = 0; i < numSamples; ++i) {
                float lvl = std::abs(buffer.getSample(0, i));
                if (numCh > 1) lvl = std::max(lvl, std::abs(buffer.getSample(1, i)));
                float target = lvl > thrLin ? 1.0f : minG;
                if (lvl > thrLin) openNow = true;
                gateEnv += (target - gateEnv) * (target > gateEnv ? 0.02f : 0.0015f);
                for (int ch = 0; ch < numCh; ++ch)
                    buffer.getWritePointer(ch)[i] *= gateEnv;
            }
            meterGateOpen.store(openNow ? 1 : 0, std::memory_order_relaxed);
        } else {
            meterGateOpen.store(1, std::memory_order_relaxed);
        }

        juce::dsp::AudioBlock<float> block(buffer);
        juce::dsp::ProcessContextReplacing<float> ctx(block);

        // 3. PRO-Q 4-BAND EQ
        if (val("eqOn") > 0.5f) {
            float maxF = (float)(sampleRate * 0.44);
            *hpf.state = *juce::dsp::IIR::Coefficients<float>::makeHighPass(sampleRate, std::clamp(val("hpf"), 20.0f, maxF));
            *mudFlt.state = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(sampleRate, 300.0f, 1.2f, juce::Decibels::decibelsToGain(val("mud")));
            *presFlt.state = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(sampleRate, 3200.0f, 1.2f, juce::Decibels::decibelsToGain(val("pres")));
            *highFlt.state = *juce::dsp::IIR::Coefficients<float>::makeHighShelf(sampleRate, std::min(9000.0f, maxF), 0.707f, juce::Decibels::decibelsToGain(val("high")));
            hpf.process(ctx);
            mudFlt.process(ctx);
            presFlt.process(ctx);
            highFlt.process(ctx);
        }

        // 4. DE-ESSER
        if (val("deEssOn") > 0.5f) {
            float maxF = (float)(sampleRate * 0.44);
            float f = std::clamp(val("deEssFreq"), 4000.0f, maxF);
            float amt = val("deEssAmt") / 100.0f;
            float cutDb = -(amt * 8.5f);
            *deEssFlt.state = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(sampleRate, f, 2.5f, juce::Decibels::decibelsToGain(cutDb));
            deEssFlt.process(ctx);
            meterDeEssGr.store(std::abs(cutDb), std::memory_order_relaxed);
        } else {
            meterDeEssGr.store(0.0f, std::memory_order_relaxed);
        }

        // 5. DYNAMICS (FET / LA-2A / CL 1B / DISTRESS / SSL)
        if (val("compOn") > 0.5f) {
            int cType = (int)val("compType");
            float thr = val("compThr");
            float rat = val("compRatio");
            float mk = juce::Decibels::decibelsToGain(val("compMakeup"));
            float atk = (cType == 0 || cType == 3) ? (zeroLat ? 1.5f : 3.5f) : (zeroLat ? 4.0f : 14.0f);
            float rel = (cType == 1) ? 260.0f : (cType == 2 ? 180.0f : 95.0f);

            float preRms = buffer.getRMSLevel(0, 0, numSamples);
            comp.setThreshold(thr);
            comp.setRatio(rat);
            comp.setAttack(atk);
            comp.setRelease(rel);
            comp.process(ctx);
            float postRms = buffer.getRMSLevel(0, 0, numSamples);
            float grDb = (preRms > 1e-5f && postRms > 1e-5f) ? std::max(0.0f, juce::Decibels::gainToDecibels(preRms / postRms)) : 0.0f;
            meterCompGr.store(grDb, std::memory_order_relaxed);
            buffer.applyGain(mk);
        } else {
            meterCompGr.store(0.0f, std::memory_order_relaxed);
        }

        // 6. ANALOG COLOR (TAPE / HEAT / AIR / CLIP)
        if (val("colorOn") > 0.5f) {
            float maxF = (float)(sampleRate * 0.44);
            float airDb = val("airGain");
            if (airDb > 0.05f) {
                *airFlt.state = *juce::dsp::IIR::Coefficients<float>::makeHighShelf(sampleRate, std::min(12000.0f, maxF), 0.7f, juce::Decibels::decibelsToGain(airDb));
                airFlt.process(ctx);
            }
            float drv = 1.0f + (val("tapeDrive") / 100.0f) * 9.0f;
            float clipG = juce::Decibels::decibelsToGain(val("clipDrive"));
            float cmix = val("colorMix") / 100.0f;
            float norm = std::tanh(drv);
            for (int ch = 0; ch < numCh; ++ch) {
                auto* d = buffer.getWritePointer(ch);
                for (int i = 0; i < numSamples; ++i) {
                    float s = d[i];
                    float sat = std::tanh(s * drv * clipG) / (norm * std::max(1.0f, clipG * 0.65f));
                    d[i] = s * (1.0f - cmix) + sat * cmix;
                }
            }
        }

        // 7. SPACE (DOUBLER / CE-1 + 201 ECHO / TAPE DELAY)
        if (val("spaceOn") > 0.5f && !dlyBufL.empty()) {
            const int bufSz = (int)dlyBufL.size();
            const float dblMix = (val("dblMix") / 100.0f) * 0.65f;
            const int dlySmp = std::clamp((int)(sampleRate * (val("dlyTime") / 1000.0f)), 64, bufSz - 64);
            const float fb = val("dlyFb") / 100.0f;
            const float dMix = (val("dlyMix") / 100.0f) * 0.75f;
            const float lfoInc = 6.2831853f * 0.9f / (float)sampleRate;

            auto* l = buffer.getWritePointer(0);
            auto* r = numCh > 1 ? buffer.getWritePointer(1) : nullptr;
            for (int i = 0; i < numSamples; ++i) {
                float inL = l[i];
                float inR = r ? r[i] : inL;

                // Stereo Haas/Chorus Doubler
                chorusPhase += lfoInc;
                if (chorusPhase > 6.2831853f) chorusPhase -= 6.2831853f;
                int cOffL = std::clamp((int)(sampleRate * (0.014f + 0.002f * std::sin(chorusPhase))), 10, bufSz - 10);
                int cOffR = std::clamp((int)(sampleRate * (0.021f - 0.002f * std::sin(chorusPhase))), 10, bufSz - 10);
                float dblL = dlyBufL[(dlyWrite - cOffL + bufSz) % bufSz];
                float dblR = dlyBufR[(dlyWrite - cOffR + bufSz) % bufSz];

                // 201 Tape Delay with LPF damping
                int rdIdx = (dlyWrite - dlySmp + bufSz) % bufSz;
                float tapL = dlyBufL[rdIdx];
                float tapR = dlyBufR[rdIdx];
                dlyLpL = dlyLpL * 0.62f + tapL * 0.38f;
                dlyLpR = dlyLpR * 0.62f + tapR * 0.38f;

                dlyBufL[dlyWrite] = std::clamp(inL + dlyLpR * fb, -1.5f, 1.5f);
                dlyBufR[dlyWrite] = std::clamp(inR + dlyLpL * fb, -1.5f, 1.5f);
                dlyWrite = (dlyWrite + 1) % bufSz;

                l[i] = inL + dblL * dblMix + dlyLpL * dMix;
                if (r) r[i] = inR + dblR * dblMix + dlyLpR * dMix;
            }
        }

        // 8. REVERB (LUMEN / GHOST) + OUTPUT LIMITER
        if (val("revOn") > 0.5f) {
            bool ghost = (int)val("revType") == 1;
            juce::dsp::Reverb::Parameters rp;
            rp.roomSize = val("revSize");
            rp.damping = ghost ? 0.78f : 0.28f;
            rp.width = 1.0f;
            rp.wetLevel = (val("revMix") / 100.0f) * 0.65f;
            rp.dryLevel = 1.0f - (val("revMix") / 100.0f) * 0.25f;
            reverb.setParameters(rp);
            reverb.process(ctx);
        }

        // Global Mix + Out Gain + Brickwall Ceiling Limiter
        const float ceilLin = juce::Decibels::decibelsToGain(val("limCeil"));
        float peakOut = 0.0f;
        for (int ch = 0; ch < numCh; ++ch) {
            auto* wet = buffer.getWritePointer(ch);
            const auto* dry = dryBuf.getReadPointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                float s = (dry[i] * (1.0f - globalMix) + wet[i] * globalMix) * outLinear;
                if (std::abs(s) > ceilLin) s = std::tanh(s / ceilLin) * ceilLin;
                wet[i] = s;
                peakOut = std::max(peakOut, std::abs(s));
            }
        }
        meterOutPeak.store(peakOut, std::memory_order_relaxed);
    }

    const juce::String getName() const override { return "VoxChain Pro"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 3.0; }
    int getNumPrograms() override { return (int)kArtistPresets.size(); }
    int getCurrentProgram() override { return currentPresetIdx; }
    void setCurrentProgram(int idx) override { applyArtistPreset(idx); }
    const juce::String getProgramName(int idx) override {
        if (idx >= 0 && idx < (int)kArtistPresets.size()) return kArtistPresets[(size_t)idx].name;
        return "Default";
    }
    void changeProgramName(int, const juce::String&) override {}
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override;
    void releaseResources() override {}
    void getStateInformation(juce::MemoryBlock& dest) override {
        if (auto xml = apvts.copyState().createXml()) copyXmlToBinary(*xml, dest);
    }
    void setStateInformation(const void* data, int size) override {
        if (auto xml = getXmlFromBinary(data, size)) apvts.replaceState(juce::ValueTree::fromXml(*xml));
    }

    juce::AudioProcessorValueTreeState apvts;
    int currentPresetIdx = 0;
    std::atomic<float> meterInPeak{0.0f}, meterOutPeak{0.0f}, meterCompGr{0.0f}, meterDeEssGr{0.0f}, meterFreq{0.0f};
    std::atomic<int> meterMidi{-1}, meterCents{0}, meterGateOpen{1};

private:
    double sampleRate = 44100.0;
    using Filter = juce::dsp::ProcessorDuplicator<juce::dsp::IIR::Filter<float>, juce::dsp::IIR::Coefficients<float>>;
    Filter hpf, mudFlt, presFlt, highFlt, airFlt, deEssFlt;
    juce::dsp::Compressor<float> comp;
    juce::dsp::Reverb reverb;
    VoxPitchShifter pitchShifter;
    std::vector<float> dlyBufL, dlyBufR;
    int dlyWrite = 0;
    float dlyLpL = 0.0f, dlyLpR = 0.0f, gateEnv = 1.0f, chorusPhase = 0.0f;
};

// ============================================================================
// 4. CUSTOM HARDWARE LOOK & FEEL (DIETER RAMS / TEENAGE ENGINEERING STYLE)
// ============================================================================
class HardwareLookAndFeel : public juce::LookAndFeel_V4 {
public:
    void drawRotarySlider(juce::Graphics& g, int x, int y, int w, int h, float sliderPos,
                          float startAngle, float endAngle, juce::Slider& slider) override {
        const float size = (float)std::min(w, h) - 6.0f;
        const float cx = x + w * 0.5f;
        const float cy = y + h * 0.5f;
        const float rOuter = size * 0.46f;
        const float rCap = size * 0.31f;
        const bool isBig = slider.getProperties().getWithDefault("isBig", false);
        const auto accent = slider.findColour(juce::Slider::rotarySliderFillColourId);

        // Dotted scale
        const int dots = isBig ? 23 : 13;
        g.setColour(isBig ? juce::Colour(0xff64748b) : juce::Colour(0x8894a3b8));
        for (int i = 0; i < dots; ++i) {
            float a = startAngle + ((float)i / (float)(dots - 1)) * (endAngle - startAngle) - juce::MathConstants<float>::halfPi;
            float dx = cx + std::cos(a) * (rOuter + 3.0f);
            float dy = cy + std::sin(a) * (rOuter + 3.0f);
            g.fillEllipse(dx - 1.2f, dy - 1.2f, 2.4f, 2.4f);
        }

        // Active value arc
        juce::Path arc;
        float curAngle = startAngle + sliderPos * (endAngle - startAngle);
        arc.addCentredArc(cx, cy, rOuter - 2.0f, rOuter - 2.0f, 0.0f, startAngle, curAngle, true);
        g.setColour(accent);
        g.strokePath(arc, juce::PathStrokeType(isBig ? 3.5f : 2.6f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

        // 3D Knob shadow & cap
        g.setColour(juce::Colour(0x40000000));
        g.fillEllipse(cx - rCap, cy - rCap + 3.0f, rCap * 2.0f, rCap * 2.0f);

        juce::ColourGradient grad(isBig ? juce::Colour(0xffffffff) : juce::Colour(0xff303642), cx, cy - rCap,
                                  isBig ? juce::Colour(0xffdfe3ea) : juce::Colour(0xff161920), cx, cy + rCap, false);
        g.setGradientFill(grad);
        g.fillEllipse(cx - rCap, cy - rCap, rCap * 2.0f, rCap * 2.0f);

        g.setColour(isBig ? juce::Colour(0xffcbd5e1) : juce::Colour(0x33ffffff));
        g.drawEllipse(cx - rCap, cy - rCap, rCap * 2.0f, rCap * 2.0f, 1.2f);

        // Indicator notch
        float ang = curAngle - juce::MathConstants<float>::halfPi;
        g.setColour(isBig ? juce::Colour(0xff181b20) : accent);
        g.drawLine(cx + std::cos(ang) * (rCap * 0.35f), cy + std::sin(ang) * (rCap * 0.35f),
                   cx + std::cos(ang) * (rCap * 0.85f), cy + std::sin(ang) * (rCap * 0.85f), isBig ? 3.2f : 2.5f);
    }

    void drawLinearSlider(juce::Graphics& g, int x, int y, int w, int h, float sliderPos,
                          float, float, const juce::Slider::SliderStyle, juce::Slider&) override {
        float cx = x + w * 0.5f;
        float railW = 8.0f;
        juce::Rectangle<float> rail(cx - railW * 0.5f, (float)y + 8.0f, railW, (float)h - 16.0f);
        g.setColour(juce::Colour(0xff151820));
        g.fillRoundedRectangle(rail, 4.0f);

        // Active orange fill from bottom
        float fillH = std::max(0.0f, rail.getBottom() - sliderPos);
        g.setColour(juce::Colour(0xffff5520));
        g.fillRoundedRectangle(rail.getX() + 1.0f, sliderPos, railW - 2.0f, fillH, 3.0f);

        // Zero center tick
        g.setColour(juce::Colour(0xff94a3b8));
        g.drawHorizontalLine((int)rail.getCentreY(), cx - 12.0f, cx + 12.0f);

        // Tactile white fader cap
        juce::Rectangle<float> thumb(cx - 14.0f, sliderPos - 8.0f, 28.0f, 16.0f);
        g.setColour(juce::Colour(0x35000000));
        g.fillRoundedRectangle(thumb.translated(0, 2), 4.0f);
        g.setColour(juce::Colour(0xfff8fafc));
        g.fillRoundedRectangle(thumb, 4.0f);
        g.setColour(juce::Colour(0xff94a3b8));
        g.drawRoundedRectangle(thumb, 4.0f, 1.2f);
        g.setColour(juce::Colour(0xff181b20));
        g.fillRect(thumb.getX() + 5.0f, thumb.getCentreY() - 1.0f, thumb.getWidth() - 10.0f, 2.0f);
    }
};

// ============================================================================
// 5. HARDWARE MODULE CARD COMPONENT
// ============================================================================
struct ModuleCard : public juce::Component {
    juce::String title, category, liveMain = "--", liveSub = "READY";
    juce::Colour bgCol, textCol, accentCol;
    float meterPct = 0.0f;
    bool isBypassed = false;

    juce::ToggleButton pwrBtn{"ON"};
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> pwrAtt;

    struct KnobItem {
        juce::Slider slider;
        juce::Label label;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> att;
    };
    std::vector<std::unique_ptr<KnobItem>> knobs;

    std::unique_ptr<juce::ComboBox> combo1, combo2;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> comboAtt1, comboAtt2;

    ModuleCard(juce::String t, juce::String c, juce::Colour bg, juce::Colour txt, juce::Colour acc,
               juce::AudioProcessorValueTreeState& apvts, const juce::String& onParam)
        : title(t), category(c), bgCol(bg), textCol(txt), accentCol(acc) {
        addAndMakeVisible(pwrBtn);
        pwrBtn.setClickingTogglesState(true);
        pwrBtn.setColour(juce::ToggleButton::tickColourId, juce::Colour(0xffa3e635));
        pwrBtn.setColour(juce::ToggleButton::textColourId, txt);
        pwrAtt = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(apvts, onParam, pwrBtn);
    }

    void addKnob(juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId, const juce::String& lbl) {
        auto k = std::make_unique<KnobItem>();
        k->slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
        k->slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 58, 14);
        k->slider.setColour(juce::Slider::rotarySliderFillColourId, accentCol);
        k->slider.setColour(juce::Slider::textBoxTextColourId, textCol);
        k->slider.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        addAndMakeVisible(k->slider);

        k->label.setText(lbl, juce::dontSendNotification);
        k->label.setJustificationType(juce::Justification::centred);
        k->label.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
        k->label.setColour(juce::Label::textColourId, textCol.withAlpha(0.75f));
        addAndMakeVisible(k->label);

        k->att = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(apvts, paramId, k->slider);
        knobs.push_back(std::move(k));
    }

    void addChoice(juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId, const juce::StringArray& items, bool second = false) {
        auto cb = std::make_unique<juce::ComboBox>();
        for (int i = 0; i < items.size(); ++i) cb->addItem(items[i], i + 1);
        cb->setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff171a21));
        cb->setColour(juce::ComboBox::textColourId, juce::Colours::white);
        cb->setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff2c313b));
        addAndMakeVisible(*cb);
        auto att = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment>(apvts, paramId, *cb);
        if (!second) { combo1 = std::move(cb); comboAtt1 = std::move(att); }
        else         { combo2 = std::move(cb); comboAtt2 = std::move(att); }
    }

    void paint(juce::Graphics& g) override {
        auto b = getLocalBounds().toFloat().reduced(2.0f);
        g.setColour(isBypassed ? bgCol.withMultipliedSaturation(0.35f).withAlpha(0.65f) : bgCol);
        g.fillRoundedRectangle(b, 16.0f);
        g.setColour(juce::Colour(0x28000000));
        g.drawRoundedRectangle(b, 16.0f, 1.5f);

        // Dark OLED Screen
        auto oled = b.reduced(10.0f).withTrimmedTop(24.0f).removeFromTop(62.0f);
        g.setColour(juce::Colour(0xff111318));
        g.fillRoundedRectangle(oled, 10.0f);
        g.setColour(juce::Colour(0xff2a2f3a));
        g.drawRoundedRectangle(oled, 10.0f, 1.5f);

        auto oledInner = oled.reduced(8.0f, 5.0f);
        auto topRow = oledInner.removeFromTop(16.0f);
        g.setColour(juce::Colours::white);
        g.setFont(juce::Font(juce::FontOptions(12.5f, juce::Font::bold)));
        g.drawText(title, topRow, juce::Justification::left);

        g.setColour(juce::Colour(0xff8b95a5));
        g.setFont(juce::Font(juce::FontOptions(9.5f, juce::Font::bold)));
        g.drawText(category, topRow, juce::Justification::right);

        oledInner.removeFromTop(2.0f);
        auto midRow = oledInner.removeFromTop(20.0f);
        g.setColour(isBypassed ? juce::Colour(0xff64748b) : accentCol);
        g.setFont(juce::Font(juce::FontOptions(16.0f, juce::Font::bold)));
        g.drawText(isBypassed ? "OFF" : liveMain, midRow, juce::Justification::left);

        g.setColour(juce::Colour(0xff94a3b8));
        g.setFont(juce::Font(juce::FontOptions(9.5f)));
        g.drawText(isBypassed ? "BYPASSED" : liveSub, midRow, juce::Justification::right);

        // OLED Meter bar
        auto bar = oledInner.removeFromBottom(4.0f);
        g.setColour(juce::Colour(0xff232834));
        g.fillRoundedRectangle(bar, 2.0f);
        if (!isBypassed && meterPct > 0.01f) {
            g.setColour(accentCol);
            g.fillRoundedRectangle(bar.withWidth(bar.getWidth() * std::clamp(meterPct, 0.0f, 1.0f)), 2.0f);
        }
    }

    void resized() override {
        auto r = getLocalBounds().reduced(10);
        pwrBtn.setBounds(r.removeFromTop(22).removeFromLeft(56));
        r.removeFromTop(68); // Space for OLED

        if (combo1 || combo2) {
            auto cr = r.removeFromTop(24);
            if (combo1 && combo2) {
                combo1->setBounds(cr.removeFromLeft(cr.getWidth() / 2 - 3));
                cr.removeFromLeft(6);
                combo2->setBounds(cr);
            } else if (combo1) {
                combo1->setBounds(cr);
            }
            r.removeFromTop(6);
        }

        const int cols = 2;
        const int rows = std::max(1, ((int)knobs.size() + 1) / 2);
        const int cw = r.getWidth() / cols;
        const int ch = r.getHeight() / rows;
        for (size_t i = 0; i < knobs.size(); ++i) {
            int col = (int)i % cols;
            int row = (int)i / cols;
            auto cell = juce::Rectangle<int>(r.getX() + col * cw, r.getY() + row * ch, cw, ch).reduced(2);
            knobs[i]->label.setBounds(cell.removeFromTop(14));
            knobs[i]->slider.setBounds(cell);
        }
    }
};

// ============================================================================
// 6. MAIN HARDWARE EDITOR (1120 x 680 px)
// ============================================================================
class VoxChainProEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit VoxChainProEditor(VoxChainProProcessor& p) : AudioProcessorEditor(&p), proc(p) {
        setLookAndFeel(&lnf);

        // Top Bar Controls
        for (size_t i = 0; i < kArtistPresets.size(); ++i)
            presetBox.addItem(kArtistPresets[i].name, (int)i + 1);
        presetBox.setSelectedId(proc.currentPresetIdx + 1, juce::dontSendNotification);
        presetBox.setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff111318));
        presetBox.setColour(juce::ComboBox::textColourId, juce::Colours::white);
        presetBox.setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff2c313b));
        presetBox.onChange = [this] {
            int id = presetBox.getSelectedId() - 1;
            if (id >= 0) proc.applyArtistPreset(id);
            repaint();
        };
        addAndMakeVisible(presetBox);

        zlBtn.setClickingTogglesState(true);
        zlBtn.setColour(juce::TextButton::buttonColourId, juce::Colour(0xffe7eaef));
        zlBtn.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff171a21));
        zlBtn.setColour(juce::TextButton::textColourOffId, juce::Colour(0xff181b20));
        zlBtn.setColour(juce::TextButton::textColourOnId, juce::Colour(0xffa3e635));
        addAndMakeVisible(zlBtn);
        zlAtt = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(proc.apvts, "zeroLatency", zlBtn);

        bypassBtn.setClickingTogglesState(true);
        bypassBtn.setColour(juce::TextButton::buttonColourId, juce::Colour(0xffe7eaef));
        bypassBtn.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xffe11d48));
        bypassBtn.setColour(juce::TextButton::textColourOffId, juce::Colour(0xff181b20));
        bypassBtn.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
        addAndMakeVisible(bypassBtn);
        bypassAtt = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(proc.apvts, "globalBypass", bypassBtn);

        // Build 8 Hardware Module Cards
        auto mkCard = [&](juce::String t, juce::String c, uint32_t bg, uint32_t txt, uint32_t acc, const char* onP) {
            auto card = std::make_unique<ModuleCard>(t, c, juce::Colour(bg), juce::Colour(txt), juce::Colour(acc), proc.apvts, onP);
            addAndMakeVisible(*card);
            return card;
        };

        auto cTune = mkCard("TUNE", "AUTO-TUNE", 0xfff8f9fb, 0xff181b20, 0xffff5520, "tuneOn");
        cTune->addChoice(proc.apvts, "tuneKey", {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"}, false);
        cTune->addChoice(proc.apvts, "tuneScale", {"Minor","Major","Chrom","Penta"}, true);
        cTune->addKnob(proc.apvts, "tuneRetune", "RETUNE MS");
        cTune->addKnob(proc.apvts, "tuneMix", "MIX %");
        cards.push_back(std::move(cTune));

        auto cGate = mkCard("NOISE RED", "GATE / CLEAN", 0xffd5dce6, 0xff181b20, 0xff16a34a, "gateOn");
        cGate->addKnob(proc.apvts, "gateThr", "GATE THR");
        cGate->addKnob(proc.apvts, "gateRed", "REDUCTION");
        cards.push_back(std::move(cGate));

        auto cEq = mkCard("PRO-Q", "4-BAND EQ", 0xff252a34, 0xfff8fafc, 0xff06b6d4, "eqOn");
        cEq->addKnob(proc.apvts, "hpf", "LOW CUT");
        cEq->addKnob(proc.apvts, "mud", "MUD 300");
        cEq->addKnob(proc.apvts, "pres", "PRES 3.2K");
        cEq->addKnob(proc.apvts, "high", "HIGH 9K");
        cards.push_back(std::move(cEq));

        auto cDeEss = mkCard("DE-ESSER", "DYN SIB", 0xfff8f9fb, 0xff181b20, 0xff0284c7, "deEssOn");
        cDeEss->addKnob(proc.apvts, "deEssFreq", "FREQ HZ");
        cDeEss->addKnob(proc.apvts, "deEssAmt", "SUPPRESS");
        cards.push_back(std::move(cDeEss));

        auto cComp = mkCard("DYNAMICS", "1176/LA2A/CL1B", 0xff1e293b, 0xfff8fafc, 0xff38bdf8, "compOn");
        cComp->addChoice(proc.apvts, "compType", {"FET 1176","LA-2A","CL 1B","DISTRESS","SSL STRIP"}, false);
        cComp->addKnob(proc.apvts, "compThr", "THRESH");
        cComp->addKnob(proc.apvts, "compRatio", "RATIO");
        cComp->addKnob(proc.apvts, "compMakeup", "OUTPUT");
        cards.push_back(std::move(cComp));

        auto cColor = mkCard("TAPE / AIR", "HEAT & CLIP", 0xffe8c97a, 0xff1b170e, 0xffff5520, "colorOn");
        cColor->addKnob(proc.apvts, "tapeDrive", "TAPE/HEAT");
        cColor->addKnob(proc.apvts, "airGain", "AIR 12K");
        cColor->addKnob(proc.apvts, "clipDrive", "CLIP DB");
        cColor->addKnob(proc.apvts, "colorMix", "MIX %");
        cards.push_back(std::move(cColor));

        auto cSpace = mkCard("201 / CE-1", "ECHO & WIDTH", 0xff4f5d33, 0xfff8fafc, 0xffa3e635, "spaceOn");
        cSpace->addKnob(proc.apvts, "dblMix", "DOUBLER");
        cSpace->addKnob(proc.apvts, "dlyTime", "TIME MS");
        cSpace->addKnob(proc.apvts, "dlyFb", "FEEDBACK");
        cSpace->addKnob(proc.apvts, "dlyMix", "ECHO MIX");
        cards.push_back(std::move(cSpace));

        auto cRev = mkCard("LUMEN / GHOST", "REV + LIM", 0xff1b1824, 0xfff8fafc, 0xffa855f7, "revOn");
        cRev->addChoice(proc.apvts, "revType", {"LUMEN BRIGHT","GHOST DARK"}, false);
        cRev->addKnob(proc.apvts, "revSize", "DECAY");
        cRev->addKnob(proc.apvts, "revMix", "REV MIX");
        cRev->addKnob(proc.apvts, "limCeil", "LIM CEIL");
        cards.push_back(std::move(cRev));

        // Master Strip Controls
        mixKnob.setSliderStyle(juce::Slider::RotaryVerticalDrag);
        mixKnob.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 18);
        mixKnob.setColour(juce::Slider::rotarySliderFillColourId, juce::Colour(0xffff5520));
        mixKnob.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xff181b20));
        mixKnob.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        mixKnob.getProperties().set("isBig", true);
        addAndMakeVisible(mixKnob);
        mixAtt = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(proc.apvts, "mix", mixKnob);

        auto setupFader = [&](juce::Slider& s, const char* id, std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment>& att) {
            s.setSliderStyle(juce::Slider::LinearVertical);
            s.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 54, 16);
            s.setColour(juce::Slider::textBoxTextColourId, juce::Colour(0xff181b20));
            s.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
            addAndMakeVisible(s);
            att = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(proc.apvts, id, s);
        };
        setupFader(inFader, "inGain", inAtt);
        setupFader(outFader, "outGain", outAtt);

        // Set size LAST so resized() lays out all 8 cards and master controls properly
        setSize(1120, 680);
        resized();
        timerCallback();
        startTimerHz(25);
    }

    ~VoxChainProEditor() override {
        stopTimer();
        setLookAndFeel(nullptr);
    }

    void paint(juce::Graphics& g) override {
        g.fillAll(juce::Colour(0xffe6e8ed));

        // Top Deck
        auto top = getLocalBounds().removeFromTop(66);
        g.setColour(juce::Colour(0xfff5f6f8));
        g.fillRect(top);
        g.setColour(juce::Colour(0xffcfd4dc));
        g.drawHorizontalLine(65, 0.0f, (float)getWidth());

        // Brand OLED
        juce::Rectangle<float> brandOled(16.0f, 11.0f, 240.0f, 44.0f);
        g.setColour(juce::Colour(0xff111318));
        g.fillRoundedRectangle(brandOled, 10.0f);
        g.setColour(juce::Colour(0xff2c313b));
        g.drawRoundedRectangle(brandOled, 10.0f, 1.5f);

        g.setColour(juce::Colour(0xffff5520));
        g.setFont(juce::Font(juce::FontOptions(22.0f, juce::Font::bold)));
        g.drawText(juce::String(proc.currentPresetIdx + 1).paddedLeft('0', 2), brandOled.reduced(12, 0).removeFromLeft(36), juce::Justification::centredLeft);

        auto brandTextRect = brandOled.withTrimmedLeft(54.0f).reduced(4.0f, 6.0f);
        g.setColour(juce::Colours::white);
        g.setFont(juce::Font(juce::FontOptions(13.0f, juce::Font::bold)));
        g.drawText("VOXCHAIN PRO", brandTextRect.removeFromTop(17.0f), juce::Justification::left);
        g.setColour(juce::Colour(0xff8b95a5));
        g.setFont(juce::Font(juce::FontOptions(9.5f)));
        g.drawText("HARDWARE VOCAL RACK - VST3", brandTextRect, juce::Justification::left);

        // Right Master Panel
        auto masterArea = getLocalBounds().removeFromRight(180).withTrimmedTop(66);
        g.setColour(juce::Colour(0xfff4f5f7));
        g.fillRect(masterArea);
        g.setColour(juce::Colour(0xffcfd4dc));
        g.drawVerticalLine(masterArea.getX(), 66.0f, (float)getHeight());

        g.setColour(juce::Colour(0xff646c79));
        g.setFont(juce::Font(juce::FontOptions(11.0f, juce::Font::bold)));
        g.drawText("MASTER I/O", masterArea.getX(), 78, masterArea.getWidth(), 18, juce::Justification::centred);
        g.drawText("GLOBAL MIX", masterArea.getX(), 224, masterArea.getWidth(), 16, juce::Justification::centred);
        g.drawText("IN", masterArea.getX() + 18, 434, 60, 16, juce::Justification::centred);
        g.drawText("OUT", masterArea.getX() + 102, 434, 60, 16, juce::Justification::centred);

        // Dual OLED VU Meter in Master Panel
        juce::Rectangle<float> vuBox((float)masterArea.getX() + 34.0f, 470.0f, 112.0f, 178.0f);
        g.setColour(juce::Colour(0xff111318));
        g.fillRoundedRectangle(vuBox, 12.0f);
        g.setColour(juce::Colour(0xff2c313b));
        g.drawRoundedRectangle(vuBox, 12.0f, 1.5f);

        auto drawMeter = [&](float xPos, float lvl, const char* lbl) {
            juce::Rectangle<float> tr(xPos, vuBox.getY() + 16.0f, 20.0f, 126.0f);
            g.setColour(juce::Colour(0xff232834));
            g.fillRoundedRectangle(tr, 4.0f);
            float h = tr.getHeight() * std::clamp(lvl * 1.15f, 0.0f, 1.0f);
            g.setColour(lvl > 0.88f ? juce::Colour(0xffff5520) : juce::Colour(0xffa3e635));
            g.fillRoundedRectangle(tr.getX(), tr.getBottom() - h, tr.getWidth(), h, 4.0f);
            g.setColour(juce::Colour(0xff94a3b8));
            g.setFont(juce::Font(juce::FontOptions(10.0f, juce::Font::bold)));
            g.drawText(lbl, (int)xPos - 4, (int)tr.getBottom() + 6, 28, 16, juce::Justification::centred);
        };
        drawMeter(vuBox.getX() + 24.0f, vuInSmooth, "IN");
        drawMeter(vuBox.getX() + 68.0f, vuOutSmooth, "OUT");
    }

    void resized() override {
        presetBox.setBounds(274, 15, 340, 36);
        zlBtn.setBounds(630, 15, 220, 36);
        bypassBtn.setBounds(864, 15, 110, 36);

        // 4 x 2 Grid for 8 Hardware Modules
        auto rack = getLocalBounds().withTrimmedTop(66).withTrimmedRight(180).reduced(12);
        const int cols = 4, rows = 2;
        const int cw = rack.getWidth() / cols;
        const int ch = rack.getHeight() / rows;
        for (size_t i = 0; i < cards.size(); ++i) {
            int col = (int)i % cols;
            int row = (int)i / cols;
            cards[i]->setBounds(rack.getX() + col * cw, rack.getY() + row * ch, cw, ch);
        }

        // Right Master Strip
        auto m = getLocalBounds().removeFromRight(180).withTrimmedTop(66);
        mixKnob.setBounds(m.getX() + 30, 102, 120, 120);
        inFader.setBounds(m.getX() + 18, 256, 60, 174);
        outFader.setBounds(m.getX() + 102, 256, 60, 174);
    }

private:
    void timerCallback() override {
        static const char* noteNames[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
        auto val = [&](const char* id) { return proc.apvts.getRawParameterValue(id)->load(); };

        bool zl = val("zeroLatency") > 0.5f;
        zlBtn.setButtonText(zl ? "ZERO LATENCY - 0.0 MS" : "HQ STUDIO - 5.8 MS");

        vuInSmooth = vuInSmooth * 0.72f + proc.meterInPeak.load(std::memory_order_relaxed) * 0.28f;
        vuOutSmooth = vuOutSmooth * 0.72f + proc.meterOutPeak.load(std::memory_order_relaxed) * 0.28f;

        if (cards.size() == 8) {
            // 0: TUNE
            cards[0]->isBypassed = val("tuneOn") < 0.5f;
            int midi = proc.meterMidi.load(std::memory_order_relaxed);
            int cents = proc.meterCents.load(std::memory_order_relaxed);
            float freq = proc.meterFreq.load(std::memory_order_relaxed);
            if (midi >= 0 && freq > 60.0f) {
                cards[0]->liveMain = juce::String(noteNames[midi % 12]) + juce::String(midi / 12 - 1);
                cards[0]->liveSub = (cents > 0 ? "+" : "") + juce::String(cents) + " ct";
                cards[0]->meterPct = std::min(1.0f, std::abs((float)cents) / 50.0f);
            } else {
                cards[0]->liveMain = "--"; cards[0]->liveSub = "WAITING"; cards[0]->meterPct = 0.0f;
            }
            // 1: GATE
            cards[1]->isBypassed = val("gateOn") < 0.5f;
            bool gOpen = proc.meterGateOpen.load(std::memory_order_relaxed) > 0;
            cards[1]->liveMain = gOpen ? "OPEN" : "GATED";
            cards[1]->liveSub = juce::String((int)val("gateThr")) + " dB";
            cards[1]->meterPct = gOpen ? 0.8f : 0.15f;
            // 2: PRO-Q
            cards[2]->isBypassed = val("eqOn") < 0.5f;
            cards[2]->liveMain = juce::String((int)val("hpf")) + " Hz";
            cards[2]->liveSub = "P:" + juce::String(val("pres"), 1) + " H:" + juce::String(val("high"), 1);
            cards[2]->meterPct = std::min(1.0f, std::abs(val("pres")) / 12.0f + 0.2f);
            // 3: DE-ESSER
            cards[3]->isBypassed = val("deEssOn") < 0.5f;
            float dgr = proc.meterDeEssGr.load(std::memory_order_relaxed);
            cards[3]->liveMain = "-" + juce::String(dgr, 1) + " dB";
            cards[3]->liveSub = juce::String(val("deEssFreq") / 1000.0f, 1) + " kHz";
            cards[3]->meterPct = std::min(1.0f, dgr / 10.0f);
            // 4: COMP
            cards[4]->isBypassed = val("compOn") < 0.5f;
            float cgr = proc.meterCompGr.load(std::memory_order_relaxed);
            cards[4]->liveMain = "-" + juce::String(cgr, 1) + " dB";
            cards[4]->liveSub = "GAIN RED";
            cards[4]->meterPct = std::min(1.0f, cgr / 14.0f);
            // 5: COLOR
            cards[5]->isBypassed = val("colorOn") < 0.5f;
            cards[5]->liveMain = juce::String((int)val("tapeDrive")) + "%";
            cards[5]->liveSub = "AIR +" + juce::String(val("airGain"), 1) + "dB";
            cards[5]->meterPct = val("tapeDrive") / 100.0f;
            // 6: SPACE
            cards[6]->isBypassed = val("spaceOn") < 0.5f;
            cards[6]->liveMain = juce::String((int)val("dlyTime")) + " ms";
            cards[6]->liveSub = "FB " + juce::String((int)val("dlyFb")) + "%";
            cards[6]->meterPct = val("dlyMix") / 100.0f;
            // 7: REVERB
            cards[7]->isBypassed = val("revOn") < 0.5f;
            cards[7]->liveMain = juce::String((int)val("revMix")) + "%";
            cards[7]->liveSub = "LIM " + juce::String(val("limCeil"), 1) + "dB";
            cards[7]->meterPct = val("revMix") / 100.0f;

            for (auto& c : cards) c->repaint();
        }
        repaint(getWidth() - 180, 460, 180, 200);
    }

    VoxChainProProcessor& proc;
    HardwareLookAndFeel lnf;
    juce::ComboBox presetBox;
    juce::TextButton zlBtn{"ZERO LATENCY - 0.0 MS"}, bypassBtn{"BYPASS"};
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> zlAtt, bypassAtt;
    std::vector<std::unique_ptr<ModuleCard>> cards;
    juce::Slider mixKnob, inFader, outFader;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAtt, inAtt, outAtt;
    float vuInSmooth = 0.0f, vuOutSmooth = 0.0f;
};

juce::AudioProcessorEditor* VoxChainProProcessor::createEditor() { return new VoxChainProEditor(*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new VoxChainProProcessor(); }
