Detectors_path=Detectors

Task1_path=Benchmark/Benchmark_Data


python $Detectors_path/binoculars_eval_length.py --test_data_path $Task1_path/Multi_LLM/multi_llms_ChatGPT_test.json,$Task1_path/Multi_LLM/multi_llms_Claude-instant_test.json,$Task1_path/Multi_LLM/multi_llms_Google-PaLM_test.json,$Task1_path/Multi_LLM/multi_llms_Llama-2-70b_test.json