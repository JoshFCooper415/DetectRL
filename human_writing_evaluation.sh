Detectors_path=

Task4_path=

cd Benchmark\Benchmark

python $Detectors_path/likelihood_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,
python $Detectors_path/rank_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,
python $Detectors_path/logRank_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,
python $Detectors_path/entropy_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,
python $Detectors_path/LRR_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,
python $Detectors_path/NPR_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,
python $Detectors_path/DetectGPT_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,
python $Detectors_path/Fast_DetectGPT_evaluation.py --test_data_path Task4_path/paraphrase_attacks_test.json,Task4_path/perturbation_attacks_test.json,Task4_path/data_mixing_test.json,

python $Detectors_path/train_roberta.py --model_name roberta-base --save_model_path roberta_base_classifier --train_data_path Task4_path/paraphrase_attacks_train.json, --test_data_path Task4_path/paraphrase_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name roberta-base --save_model_path roberta_base_classifier --train_data_path Task4_path/perturbation_attacks_train.json, --test_data_path Task4_path/perturbation_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name roberta-base --save_model_path roberta_base_classifier --train_data_path Task4_path/data_mixing_train.json --test_data_path Task4_path/data_mixing_test.json,

python $Detectors_path/train_roberta.py --model_name roberta-large --save_model_path roberta_large_classifier --train_data_path Task4_path/paraphrase_attacks_train.json, --test_data_path Task4_path/paraphrase_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name roberta-large --save_model_path roberta_large_classifier --train_data_path Task4_path/perturbation_attacks_train.json, --test_data_path Task4_path/perturbation_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name roberta-large --save_model_path roberta_large_classifier --train_data_path Task4_path/data_mixing_train.json --test_data_path Task4_path/data_mixing_test.json,

python $Detectors_path/train_roberta.py --model_name FacebookAI/xlm-roberta-base --save_model_path xlm_roberta_base_classifier --train_data_path Task4_path/paraphrase_attacks_train.json, --test_data_path Task4_path/paraphrase_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name FacebookAI/xlm-roberta-base --save_model_path xlm_roberta_base_classifier --train_data_path Task4_path/perturbation_attacks_train.json, --test_data_path Task4_path/perturbation_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name FacebookAI/xlm-roberta-base --save_model_path xlm_roberta_base_classifier --train_data_path Task4_path/data_mixing_train.json --test_data_path Task4_path/data_mixing_test.json,

python $Detectors_path/train_roberta.py --model_name FacebookAI/xlm-roberta-large --save_model_path xlm_roberta_large_classifier --train_data_path Task4_path/paraphrase_attacks_train.json, --test_data_path Task4_path/paraphrase_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name FacebookAI/xlm-roberta-large --save_model_path xlm_roberta_large_classifier --train_data_path Task4_path/perturbation_attacks_train.json, --test_data_path Task4_path/perturbation_attacks_test.json,
python $Detectors_path/train_roberta.py --model_name FacebookAI/xlm-roberta-large --save_model_path xlm_roberta_large_classifier --train_data_path Task4_path/data_mixing_train.json --test_data_path Task4_path/data_mixing_test.json,

