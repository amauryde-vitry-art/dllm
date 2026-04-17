"""
python -u examples/llada/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm
import dllm







def CreateBaseSampleWithHistory(messages: list[list[dict[str, str]]], config: SamplerConfig, Script:ScriptArguments) -> dllm.core.samplers.BaseSamplerOutputCompleteHistory:
    parser = transformers.HfArgumentParser((Script, config))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.core.samplers.MDLMSamplerWithCompleteHistory(model=model, tokenizer=tokenizer)
    terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)
    inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

    outputs = sampler.sample(inputs, sampler_config, return_dict=True)
    sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)
    

    print(sequences)
    print('-----------------------')
    print(dllm.utils.sample_trim(tokenizer, outputs.histories_x[-1].tolist(), inputs))

    if script_args.visualize:
        terminal_visualizer.visualize(outputs.histories_x, rich=True)
    
    return outputs





