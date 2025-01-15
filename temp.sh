
Detectors_path=Detectors

Data_path=Benchmark/Benchmark_Data
Task1_path=Benchmark/Tasks/Task1
Task2_path=Benchmark/Tasks/Task2
Task3_path=Benchmark/Tasks/Task3
Task4_path=Benchmark/Tasks/Task4


# Multi Domains
printf '\n\nSTARTING MULTI-DOMAIN EVALUATION\n\n'
python $Detectors_path/binoculars_evaluation.py --test_data_path $Task1_path/multi_domains_arxiv_test.json,$Task1_path/multi_domains_writing_prompt_test.json,$Task1_path/multi_domains_xsum_test.json,$Task1_path/multi_domains_yelp_review_test.json

# Multi LLMs
printf '\n\nSTARTING MULTI-LLM EVALUATION\n\n'
#python $Detectors_path/binoculars_evaluation.py --test_data_path $Task1_path/multi_llms_ChatGPT_test.json,$Task1_path/multi_llms_Claude-instant_test.json,$Task1_path/multi_llms_Google-PaLM_test.json,$Task1_path/multi_llms_Llama-2-70b_test.json,

# Multi Attacks
printf '\n\nSTARTING MULTI-ATTACK EVALUATION\n\n'
#python $Detectors_path/binoculars_evaluation.py --test_data_path Benchmark/Benchmark_Data/Direct_Prompt/direct_prompt_test.json,$Task1_path/prompt_attacks_llm_test.json,$Task1_path/paraphrase_attacks_llm_test.json,$Task1_path/perturbation_attacks_llm_test.json,$Task1_path/data_mixing_attacks_test.json,


# Domains Generalization
printf '\n\nSTARTING DOMAIN GENERALIZATION EVALUATION\n\n'
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_domains_arxiv_test.json, --transfer_data_path $Task1_path/multi_domains_xsum_test.json,$Task1_path/multi_domains_writing_prompt_test.json,$Task1_path/multi_domains_yelp_review_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_domains_xsum_test.json, --transfer_data_path $Task1_path/multi_domains_arxiv_test.json,$Task1_path/multi_domains_writing_prompt_test.json,$Task1_path/multi_domains_yelp_review_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_domains_writing_prompt_test.json, --transfer_data_path $Task1_path/multi_domains_arxiv_test.json,$Task1_path/multi_domains_xsum_test.json,$Task1_path/multi_domains_yelp_review_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_domains_yelp_review_test.json, --transfer_data_path $Task1_path/multi_domains_arxiv_test.json,$Task1_path/multi_domains_xsum_test.json,$Task1_path/multi_domains_writing_prompt_test.json


# LLM Generalization
printf '\n\nSTARTING LLM GENERALIZATION EVALUATION\n\n'
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_llms_ChatGPT_test.json, --transfer_data_path $Task1_path/multi_llms_Claude-instant_test.json,$Task1_path/multi_llms_Google-PaLM_test.json,$Task1_path/multi_llms_Llama-2-70b_test.json
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_llms_Claude-instant_test.json, --transfer_data_path $Task1_path/multi_llms_ChatGPT_test.json,$Task1_path/multi_llms_Google-PaLM_test.json,$Task1_path/multi_llms_Llama-2-70b_test.json
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_llms_Google-PaLM_test.json, --transfer_data_path $Task1_path/multi_llms_ChatGPT_test.json,$Task1_path/multi_llms_Claude-instant_test.json,$Task1_path/multi_llms_Llama-2-70b_test.json
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/multi_llms_Llama-2-70b_test.json, --transfer_data_path $Task1_path/multi_llms_ChatGPT_test.json,$Task1_path/multi_llms_Claude-instant_test.json,$Task1_path/multi_llms_Google-PaLM_test.json

# Attack Generalization
printf '\n\nSTARTING ATTACK GENERALIZATION EVALUATION\n\n'
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path Benchmark/Benchmark_Data/Direct_Prompt/direct_prompt_test.json, --transfer_data_path $Task1_path/prompt_attacks_llm_test.json,$Task1_path/paraphrase_attacks_llm_test
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/prompt_attacks_llm_test.json, --transfer_data_path Benchmark/Benchmark_Data/Direct_Prompt/direct_prompt_test.json,$Task1_path/paraphrase_attacks_llm_test.json,$Task1_path/perturbation_attacks_llm_test.json,$Task1_path/data_mixing_attacks_test.json
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/paraphrase_attacks_llm_test.json, --transfer_data_path Benchmark/Benchmark_Data/Direct_Prompt/direct_prompt_test.json,$Task1_path/prompt_attacks_llm_test.json,$Task1_path/perturbation_attacks_llm_test.json,$Task1_path/data_mixing_attacks_test.json
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/perturbation_attacks_llm_test.json, --transfer_data_path Benchmark/Benchmark_Data/Direct_Prompt/direct_prompt_test.json,$Task1_path/prompt_attacks_llm_test.json,$Task1_path/paraphrase_attacks_llm_test.json,$Task1_path/data_mixing_attacks_test.json
#python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task1_path/data_mixing_attacks_test.json, --transfer_data_path Benchmark/Benchmark_Data/Direct_Prompt/direct_prompt_test.json,$Task1_path/prompt_attacks_llm_test.json,$Task1_path/paraphrase_attacks_llm_test.json,$Task1_path/perturbation_attacks_llm_test.json

# Human Writing
#printf '\n\nSTARTING HUMAN WRITING EVALUATION\n\n'
#python $Detectors_path/binoculars_evaluation.py --test_data_path $Task4_path/paraphrase_attacks_human_test.json,$Task4_path/perturbation_attacks_human_test.json,$Task4_path/data_mixing_attacks_test.json,$Task4_path/direct_prompt_test.json

#python $Detectors_path/binoculars_evaluation.py --test_data_path $Task3_path/cross_length_20_test.json,$Task3_path/cross_length_40_test.json,$Task3_path/cross_length_60_test.json,$Task3_path/cross_length_80_test.json,$Task3_path/cross_length_100_test.json,$Task3_path/cross_length_120_test.json,$Task3_path/cross_length_140_test.json,$Task3_path/cross_length_160_test.json,$Task3_path/cross_length_180_test.json,$Task3_path/cross_length_200_test.json,$Task3_path/cross_length_220_test.json,$Task3_path/cross_length_240_test.json,$Task3_path/cross_length_260_test.json,$Task3_path/cross_length_280_test.json,$Task3_path/cross_length_300_test.json,$Task3_path/cross_length_320_test.json,$Task3_path/cross_length_340_test.json,$Task3_path/cross_length_360_test.json

python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_20_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_40_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_60_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_80_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_100_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_120_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_140_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_160_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_180_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_200_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_220_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_240_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_260_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_280_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_300_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_320_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_340_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json
python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_360_test.json, --transfer_data_path $Task3_path/cross_length_180_test.json

python $Detectors_path/zero_shot_transfer_evaluation.py --test_data_path $Task3_path/cross_length_180_test.json, --transfer_data_path $Task3_path/cross_length_20_test.json,$Task3_path/cross_length_40_test.json,$Task3_path/cross_length_60_test.json,$Task3_path/cross_length_80_test.json,$Task3_path/cross_length_100_test.json,$Task3_path/cross_length_120_test.json,$Task3_path/cross_length_140_test.json,$Task3_path/cross_length_160_test.json,$Task3_path/cross_length_180_test.json,$Task3_path/cross_length_200_test.json,$Task3_path/cross_length_220_test.json,$Task3_path/cross_length_240_test.json,$Task3_path/cross_length_260_test.json,$Task3_path/cross_length_280_test.json,$Task3_path/cross_length_300_test.json,$Task3_path/cross_length_320_test.json,$Task3_path/cross_length_340_test.json,$Task3_path/cross_length_360_test.json