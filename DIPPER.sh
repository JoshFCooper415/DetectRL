Data_path=
Data_Generation_path=

cd Benchmark\Benchmark

# GPT-3.5-turbo
python $Data_Generation_path/DIPPER.py --llm_types ChatGPT --input_path $Data_path/arxiv_2800.json --output_path $Data_path/arxiv_2800.json
python $Data_Generation_path/DIPPER.py --llm_types ChatGPT --input_path $Data_path/xsum_2800.json --output_path $Data_path/xsum_2800.json
python $Data_Generation_path/DIPPER.py --llm_types ChatGPT --input_path $Data_path/writing_prompt_2800.json --output_path $Data_path/writing_prompt_2800.json
python $Data_Generation_path/DIPPER.py --llm_types ChatGPT --input_path $Data_path/yelp_review_2800.json --output_path $Data_path/yelp_review_2800.json

# Claude-instant
python $Data_Generation_path/DIPPER.py --llm_types Claude-instant --input_path $Data_path/arxiv_2800.json --output_path $Data_path/arxiv_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Claude-instant --input_path $Data_path/xsum_2800.json --output_path $Data_path/xsum_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Claude-instant --input_path $Data_path/writing_prompt_2800.json --output_path $Data_path/writing_prompt_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Claude-instant --input_path $Data_path/yelp_review_2800.json --output_path $Data_path/yelp_review_2800.json

# Google-PaLM
python $Data_Generation_path/DIPPER.py --llm_types Google-PaLM --input_path $Data_path/arxiv_2800.json --output_path $Data_path/arxiv_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Google-PaLM --input_path $Data_path/xsum_2800.json --output_path $Data_path/xsum_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Google-PaLM --input_path $Data_path/writing_prompt_2800.json --output_path $Data_path/writing_prompt_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Google-PaLM --input_path $Data_path/yelp_review_2800.json --output_path $Data_path/yelp_review_2800.json

# Llama-2-70b
python $Data_Generation_path/DIPPER.py --llm_types Llama-2-70b --input_path $Data_path/arxiv_2800.json --output_path $Data_path/arxiv_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Llama-2-70b --input_path $Data_path/xsum_2800.json --output_path $Data_path/xsum_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Llama-2-70b --input_path $Data_path/writing_prompt_2800.json --output_path $Data_path/writing_prompt_2800.json
python $Data_Generation_path/DIPPER.py --llm_types Llama-2-70b --input_path $Data_path/yelp_review_2800.json --output_path $Data_path/yelp_review_2800.json