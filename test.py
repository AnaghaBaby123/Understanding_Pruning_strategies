test_dspath = r'/home/anagha/Documents/MAI/ACV/Portfolio2/Traffic/Data/Test'
import pandas as pd
import os
import shutil
#arranging test data into folders based on ClassId
test_data = pd.read_csv(r'/home/anagha/Documents/MAI/ACV/Portfolio2/Traffic/Data/Test.csv')
test_data = test_data[['ClassId', 'Path']]
for fname in os.listdir(test_dspath):
    if fname.endswith('.png'):
        matching = test_data[test_data['Path'].str.endswith(fname)]
        class_id = matching.iloc[0]['ClassId']
        os.makedirs(os.path.join(test_dspath, str(class_id)), exist_ok=True)
        shutil.move(os.path.join(test_dspath, fname), os.path.join(test_dspath, str(class_id)))
