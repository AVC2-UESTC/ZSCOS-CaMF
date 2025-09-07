## embed.py



### PatchEmbed

**Architecture**

$$
{\rm Conv} \rightarrow {\rm Flatten} \rightarrow {\rm norm}
$$










## attention.py

### MultiheadAttention

```python
identity + dropout_layer(proj_drop(Attention(input)))
```



### WindowAttention

$$
{\rm Attention}(input)
$$



























