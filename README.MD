# tinyLM

From scratch implementation of a 52M parameter Transformer, following the Vaswani et al. paper. The training will be focused on English-Spanish translation, using 142k+ sentence pairs (to be expanded).

![Demo](assets/tiny_lm_demo.gif)

Originally, I was going to code it all by hand (no autograd, no nn.Modules, etc.) as I have done with other projects, but I realized that in order to maximize my learning, I'd just use all the built-in PyTorch utils.

## Benchmark

**Task:** English → Spanish translation  
**Metric:** SacreBLEU (BLEU‑4)  
**Decoding:** greedy  
**Checkpoint:** `tinyLM_bs16_lr1e-4_layers6.pt`  
**Test subset:** 200 sentences from the held‑out split

| Metric |     Score |
| ------ | --------: |
| BLEU‑4 | **19.49** |

So far, not too shabby considered that in Vaswani et al. their highest BLEU was 26.4. They could have been using a different metric (not SacreBLEU), but still interesting. Will try to get this as high as possible with the next couple training runs

Run it yourself:

```bash
python eval_bleu.py --ckpt weights/tinyLM_bs16_lr1e-4_layers6.pt --limit 200
```

Notes:

- SacreBLEU is the standard BLEU implementation.
- Scores vary by checkpoint, decoding strategy, and test size.
