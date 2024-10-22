import torch
import marlin
import time

from gptq import *
from quant import *

DEV = torch.device('cuda:0')

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_llama(name):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM, AutoTokenizer
    model = LlamaForCausalLM.from_pretrained(name, torch_dtype='auto')
    tokenizer = AutoTokenizer.from_pretrained(name)
    model.config.pretraining_tp = 1
    model.seqlen = 4096 
    return model, tokenizer

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs['attention_mask'])
            position_ids.append(kwargs['position_ids'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            if model.config.num_attention_heads != model.config.num_key_value_heads and args.skip_gq:
                names.remove('self_attn.k_proj')
                names.remove('self_attn.v_proj')

            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(args.wbits)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                res = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, clip=not args.no_clip, baseline=args.nearest
                )
                res = list(res)
                res[0] = res[0].cpu()
                res[1] = res[1].cpu()
                quantizers['model.layers.%d.%s' % (i, name)] = res

        for j in range(args.nsamples):
            inps[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return quantizers

def llama_pack(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    marlin.replace_linear_3bitFaster(model, lambda n: n in quantizers, groupsize=args.groupsize)
    qlayers = find_layers(model, [marlin.Layer3bitFaster])
    print('Packing ...')
    for name in qlayers:
        print(name)
        qlayers[name].pack(layers[name].to(DEV), quantizers[name][0].to(DEV))
        qlayers[name].cpu()
        quantizers[name][0].cpu()
        layers[name].cpu()
    print('Done.')
    return model

def test_latency(model, tokenizer):
    gen_length = 20

    with torch.inference_mode():
        for i in range(gen_length):
            torch.cuda.synchronize()
            t_st = time.perf_counter()

            if i == 0:
                inputs = torch.as_tensor([input_ids], device=device)
            else:
                inputs = torch.as_tensor([[token]], device=device)
            out = model(inputs, start_pos=start_pos)
            start_pos += out.shape[1]

            torch.cuda.synchronize()
            t_ed = time.perf_counter()
            time_lis.append(t_ed - t_st)
            token = out[:, -1].max(1)[1].unsqueeze(1)
            if args.verbose:
                print(i, np.median(time_lis))

    print(f"Speed: {1 / np.median(time_lis)} tokens per second.")

    
    latency = end_time - start_time
    print()

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        '--dataset', type=str, default='red', choices=['red'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.1,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[3, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=64, choices=[64],
        help='Groupsize to use for quantization; default is 128.'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--no_clip', action='store_true',
        help='Whether to skip hessian based grid clipping when using groups.'
    )
    parser.add_argument(
        '--skip_gq', action='store_true',
        help='Whether to skip quantizing group keys and values for the 70B model with group-query attention.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Whether and where to save the quantized model.'
    )

    args = parser.parse_args()

    if args.nearest:
        args.nsamples = 0

    model,tokenizer = get_llama(args.model)


    if args.wbits < 16:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    
    test_latency(model,tokenizer)

    if args.save:
        args.save += '.marlin3bit'
        if args.groupsize != -1:
            args.save += '.g%d' % args.groupsize
        llama_pack(model, quantizers)
        torch.save(model.state_dict(), args.save)

