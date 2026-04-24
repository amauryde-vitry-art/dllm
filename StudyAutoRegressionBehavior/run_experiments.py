from GenerateWithMDLMSampler import CreateBaseSampleWithHistory
import dllm
from dataclasses import dataclass
from GetInfoFromBaseSamplerOutput import getH, getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getNumTransferTokens, getUnmaskLogProbs
from PlotResults import PlotlyH, plotH, plotLogProbs, plotMasks, plotChanges, plotLevenshtein, plotAttentionMask, plotNumTransferredTokens, plotUnmaskLogProbs, PlotlyLogProbs, PlotlyUnmaskLogProbs, PlotlyChanges , getTxt


def run_experiment_MDML(model_cfg, messages, SamplerConfig):
    @dataclass
    class ScriptArguments:
        model_name_or_path: str = model_cfg['path']
        seed: int = 42
        visualize: bool = False

        def __post_init__(self):
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )

    outputs, tokenizer = CreateBaseSampleWithHistory(dllm.core.samplers.MDLMSamplerWithCompleteHistory,messages, SamplerConfig, ScriptArguments)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)
    attention_masks = outputs.attention_mask.detach().cpu().numpy()


    getTxt(Proposed_sequences, model_cfg['dir'])
    getTxt(Generated_sequences, model_cfg['dir'], proposed_sequence=False)

    plotLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    plotMasks(masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, remasking=False)
    plotChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    plotLevenshtein(levenshtein, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotAttentionMask(attention_masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], )


    PlotlyLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences, masks,)
    PlotlyUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks,)
    PlotlyChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences,)



def run_experiment_MDML_remasking(model_cfg, messages, SamplerConfig):
    @dataclass
    class ScriptArguments:
        model_name_or_path: str = model_cfg['path']
        seed: int = 42
        visualize: bool = False

        def __post_init__(self):
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )
    outputs, tokenizer = CreateBaseSampleWithHistory(dllm.core.samplers.MDLMSamplerRemaskingWithCompleteHistory, messages, SamplerConfig, ScriptArguments)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    remasks = getEachStepMask(outputs, remask=True)
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)
    attention_masks = outputs.attention_mask.detach().cpu().numpy()
    H = getH(outputs)
    num_transferred_tokens = getNumTransferTokens(outputs)

    getTxt(Proposed_sequences, model_cfg['dir'])
    getTxt(Generated_sequences, model_cfg['dir'], proposed_sequence=False)

    plotLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'] + '_REMASKING', outputs.block_size, Generated_sequences, masks, remasks)
    plotMasks(masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'] + '_REMASKING', outputs.block_size, Generated_sequences, remasking=False)
    plotMasks(remasks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Generated_sequences, remasking=True)
    plotChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Generated_sequences, masks, remasks)
    plotLevenshtein(levenshtein, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Generated_sequences)
    plotAttentionMask(attention_masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', )
    plotH(H, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Generated_sequences, masks, remasks)
    plotNumTransferredTokens(num_transferred_tokens, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING')

    PlotlyLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Proposed_sequences, masks,remasks)
    PlotlyUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Generated_sequences, masks, remasks)
    PlotlyChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Proposed_sequences,)
    PlotlyH(H, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir']+ '_REMASKING', outputs.block_size, Generated_sequences, masks, remasks)

