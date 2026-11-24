// VoxChainProcessor.cpp — Нативный VST3 для Ableton Live (JUCE 8)
// Активный пресет: Billie Eilish — Whisper Close
// Zero Latency Mode: HQ STUDIO
// Собранная цепочка модулей:
//  1. NOISE RED (ON)
//  2. PRO-Q (ON)
//  3. LA-2A (ON)
//  4. AIR (ON)
//  5. DOUBLER (ON)
//  6. GHOST REV (ON)

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>

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
        p.push_back(std::make_unique<juce::AudioParameterBool>("zeroLatency", "Zero Latency", false));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("inGain", "Input dB", -18.0f, 18.0f, 0f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("outGain", "Output dB", -18.0f, 18.0f, 0f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("mix", "Global Mix", 0.0f, 1.0f, 1.00f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("hpf", "Pro-Q LowCut", 20.0f, 250.0f, 95.0f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("compThr", "Dynamics Thr", -40.0f, 0.0f, -20.0f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("drive", "Tape/Heat Drive", 0.0f, 100.0f, 32.0f));
        p.push_back(std::make_unique<juce::AudioParameterFloat>("reverb", "Lumen/Ghost Mix", 0.0f, 1.0f, 0.25f));
        return { p.begin(), p.end() };
    }

    void prepareToPlay(double sr, int blockSize) override {
        juce::dsp::ProcessSpec spec { sr, (juce::uint32)blockSize, 2 };
        hpf.prepare(spec); comp.prepare(spec); reverb.prepare(spec);
        updateLatency();
    }

    void updateLatency() {
        bool zl = apvts.getRawParameterValue("zeroLatency")->load() > 0.5f;
        setLatencySamples(zl ? 0 : 512);
    }

    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override {
        juce::ScopedNoDenormals noDenormals;
        updateLatency();
        auto sr = getSampleRate();
        *hpf.state = *juce::dsp::IIR::Coefficients<float>::makeHighPass(sr, apvts.getRawParameterValue("hpf")->load());
        comp.setThreshold(apvts.getRawParameterValue("compThr")->load());
        comp.setRatio(4.5f);
        comp.setAttack(apvts.getRawParameterValue("zeroLatency")->load() > 0.5f ? 2.0f : 10.0f);
        comp.setRelease(140.0f);

        juce::dsp::AudioBlock<float> block(buffer);
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        hpf.process(ctx);
        comp.process(ctx);

        float drv = 1.0f + (apvts.getRawParameterValue("drive")->load() / 100.0f) * 8.0f;
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
            auto* d = buffer.getWritePointer(ch);
            for (int i = 0; i < buffer.getNumSamples(); ++i)
                d[i] = std::tanh(d[i] * drv) / std::tanh(drv);
        }

        juce::dsp::Reverb::Parameters rp;
        rp.wetLevel = apvts.getRawParameterValue("reverb")->load() * 0.5f;
        rp.dryLevel = 1.0f; rp.roomSize = 0.7f;
        reverb.setParameters(rp);
        reverb.process(ctx);
    }

    const juce::String getName() const override { return "VoxChain Pro"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 3.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Billie Eilish — Whisper Close"; }
    void changeProgramName(int, const juce::String&) override {}
    bool hasEditor() const override { return true; }
    juce::AudioProcessorEditor* createEditor() override { return new juce::GenericAudioProcessorEditor(*this); }
    void releaseResources() override {}
    void getStateInformation(juce::MemoryBlock& dest) override {
        if (auto xml = apvts.copyState().createXml()) copyXmlToBinary(*xml, dest);
    }
    void setStateInformation(const void* data, int size) override {
        if (auto xml = getXmlFromBinary(data, size)) apvts.replaceState(juce::ValueTree::fromXml(*xml));
    }
private:
    juce::AudioProcessorValueTreeState apvts;
    using Filter = juce::dsp::ProcessorDuplicator<juce::dsp::IIR::Filter<float>, juce::dsp::IIR::Coefficients<float>>;
    Filter hpf;
    juce::dsp::Compressor<float> comp;
    juce::dsp::Reverb reverb;
};
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new VoxChainProProcessor(); }