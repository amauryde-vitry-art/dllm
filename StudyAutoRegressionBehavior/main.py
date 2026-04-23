from GenerateWithMDLMSampler import CreateBaseSampleWithHistory
import dllm
from dataclasses import dataclass
from GetInfoFromBaseSamplerOutput import getLogProbs, getEachStepGeneratedSequence, getEachStepProposedSequence, getEachStepMask, getEachStepChange, getLevenshtein, getUnmaskLogProbs
from PlotResults import plotLogProbs, plotMasks, plotChanges, plotLevenshtein, plotAttentionMask, plotUnmaskLogProbs, PlotlyLogProbs, PlotlyUnmaskLogProbs, PlotlyChanges , getTxt


import sys
sys.argv = [sys.argv[0]] # Cette ligne "vide" les arguments du terminal



@dataclass
class SamplerConfig(dllm.core.samplers.MDLMSamplerConfig):
    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"


messages = [
    # 1. CODE : Test de la logique et de la syntaxe
    [{"role": "user", "content": "Écris une fonction Python nommée `fibonacci` qui génère les n premiers nombres de la suite. Inclut des commentaires ligne par ligne et un exemple d'utilisation."}],
    
    # 2. CULTURE G : Test de précision factuelle (Noms, dates, lieux)
    [{"role": "user", "content": "Explique brièvement ce qu'était la Révolution industrielle, en citant sa période de début, le pays où elle a commencé et deux inventions majeures de cette époque."}],
    
    # 3. HISTOIRE : Test de créativité et de cohérence narrative (longue traîne)
    [{"role": "user", "content": "Raconte une histoire courte (environ 100 mots) sur un astronaute qui découvre une bibliothèque ancienne sur une planète déserte. Le ton doit être mystérieux et la fin surprenante."}]
]


MODELS_TO_TEST = [
    # {
    #     "name": "ModernBERT",
    #     "path": "dllm-hub/ModernBERT-large-chat-v0.1",
    #     "dir": "ModernBERT"
    # },
    {
        "name": "LLaDa",
        "path": "GSAI-ML/LLaDA-8B-Instruct", 
        "dir": "LLaDa_test_remasking"
    },
    # {
    #     "name": "Qwen-MDLM",
    #     "path": "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
    #     "dir": "Qwen_MDLM"
    # },
    # {
    #     "name": "a2d",
    #     "path":  "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    #     "dir": "a2d"
        
    # },

    # {
    #     "name": "Dream",
    #     "path": "Dream-org/Dream-v0-Instruct-7B",
    #     "dir": "Dream"
        
    # },

    # {
    #     "name": "editflow",
    #     "path": "./.models/editflow/ModernBERT-large/alpaca/checkpoint-final",
    #     "dir": "editflow"
        
    # },
    # {
    #     "name": "LargeBERT",
    #     "path": "dllm-hub/ModernBERT-large-chat-v0.1",
    #     "dir": "LargeBERT"
        
    # },

    # {
    #     "name": "qwen3MDML",
    #     "path": "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
    #     "dir": "qwen3MDML"
        
    # },

    # {
    #     "name": "LLaDA2",
    #     "path": " inclusionAI/LLaDA2.0-mini",
    #     "dir": "LLaDA2"
        
    # }


]

def run_experiment(model_cfg, messages):
    @dataclass
    class ScriptArguments:
        model_name_or_path: str = model_cfg['path']
        seed: int = 42
        visualize: bool = False

        def __post_init__(self):
            self.model_name_or_path = dllm.utils.resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )

    outputs, tokenizer = CreateBaseSampleWithHistory(messages, SamplerConfig, ScriptArguments)
    logprobs = getLogProbs(outputs)
    masks = getEachStepMask(outputs)
    remasks = getEachStepMask(outputs, remask=True)
    Generated_sequences = getEachStepGeneratedSequence(outputs, tokenizer)
    Proposed_sequences = getEachStepProposedSequence(outputs, tokenizer)
    changes = getEachStepChange(outputs)
    levenshtein = getLevenshtein(Proposed_sequences)
    unmaskLogProbs = getUnmaskLogProbs(outputs)

    attention_masks = outputs.attention_mask.detach().cpu().numpy()


    getTxt(Proposed_sequences, model_cfg['dir'])
    getTxt(Generated_sequences, model_cfg['dir'], proposed_sequence=False)

    plotLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks, remasks)
    plotMasks(masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks, remasks)
    plotLevenshtein(levenshtein, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences)
    plotAttentionMask(attention_masks, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], )
    
    PlotlyLogProbs(logprobs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences, masks, remasks)
    PlotlyUnmaskLogProbs(unmaskLogProbs, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Generated_sequences, masks, remasks)
    PlotlyChanges(changes, [f'{model_cfg['name']} - EX1', f'{model_cfg['name']} - EX2', f'{model_cfg['name']} - EX3'], model_cfg['dir'], outputs.block_size, Proposed_sequences,)

def main():
    for model_cfg in MODELS_TO_TEST:
        try:
            run_experiment(model_cfg, messages)
        except Exception as e:
            print(f"❌ Error with model {model_cfg['name']}: {e}")
            continue



if __name__ == "__main__":
    main()
