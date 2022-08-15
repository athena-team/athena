# Language model
We implemented three language models for training and decoding, you can choose ngram, rnn lm and transformer lm. Language model decoding now only supports the form of "attention_rescoring". If you want to use the corresponding language model to decode, you must set "decoder_type" to "attention_rescoring", and specify "lm_type" to one of ["rnn", "transformer" , "ngram"], "lm_path" as the corresponding training json configuration and rescoring "lm_weight". For ngram language model, "lm_path" needs to be specified as the ngram language model path instead of configuration path.

[reference configuration](examples/asr/aishell/configs/mtl_transformer_sp_fbank80_decode_transformer_lm.json)

```json
  "inference_config":{
    "decoder_type":"attention_rescoring",
    "model_avg_num":10,
    "beam_size": 20,
    "at_weight": 1.0,
    "ctc_weight":1.2,
    "lm_type": "transformer",
    "lm_weight":0.5,
    "lm_path": "examples/asr/aishell/configs/transformer_lm.json",
    "predict_path": "examples/asr/aishell/result/mtl_transformer_attention_rescoring_ctc1_2_lm0_5.result",
    "label_path": "examples/asr/aishell/result/mtl_transformer_attention_rescoring_ctc1_2_lm0_5.label"
  },
```