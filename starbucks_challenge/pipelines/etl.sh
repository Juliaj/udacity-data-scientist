# generate interim datasets
python -m data_processing.profile ../data/0_raw/profile.json ../data/1_interim/profile.pkl
python -m data_processing.portfolio ../data/0_raw/portfolio.json ../data/1_interim/portfolio.pkl 
python -m data_processing.transcript ../data/0_raw/transcript.json ../data/1_interim/transcript.pkl

# generate combine datasets
python -m data_processing.offer_response ../data/1_interim/transcript.pkl ../data/1_interim/profile.pkl ../data/1_interim/portfolio.pkl ../data/1_interim/offer_response.pkl

# model building 
python -m offer_response.model ../data/1_interim/offer_response.pkl ../output/models/offer_response.pkl
