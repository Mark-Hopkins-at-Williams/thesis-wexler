import json

BASE_DIR = '/mnt/storage/swexler/thesis-wexler/models'
EXP_DIR1 = 'french-training-v5'
EXP_DIR2 = 'french-training-v15'
EXP_DIR3 = 'french-training-v16'
EXP_DIR4 = 'french-training-v23'

with open(f'{BASE_DIR}/{EXP_DIR1}/translations.json') as reader:
    data1 = json.load(reader)
with open(f'{BASE_DIR}/{EXP_DIR2}/translations.json') as reader:
    data2 = json.load(reader)
with open(f'{BASE_DIR}/{EXP_DIR3}/translations.json') as reader:
    data3 = json.load(reader)
with open(f'{BASE_DIR}/{EXP_DIR4}/translations.json') as reader:
    data4 = json.load(reader)


for i in range(7):
    print('\nbpe:')
    print(data1['eng_Latn->fra_Latn'][i])

    print('\nsmall vocab:')
    print(data2['eng_Latn->fra_Latn'][i])
    print('\nlarger vocab:')
    print(data3['eng_Latn->fra_Latn'][i])
    
    print('\nseparate src/tgt vocab:')
    print(data4['eng_Latn->fra_Latn'][i])