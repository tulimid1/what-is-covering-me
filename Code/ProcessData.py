# import libraries 
import pandas as pd

from sklearn.preprocessing import StandardScaler

class ProcessForestData():

    data_folder_path = ''
    label_colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
    classifier_undummy = ['Spruce/Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwod/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
    topFigFolderName = 'FinalFigures'

    def __init__(self):
        pass  

    # Data processing
    def load_data(self, data_folder_path=data_folder_path, sub_data_section='_tiny', perform_scale=False):

        X_train = pd.read_csv(data_folder_path+'/X_train'+sub_data_section+'.csv').to_numpy()
        X_test =  pd.read_csv(data_folder_path+'/X_test'+sub_data_section+'.csv').to_numpy()

        if perform_scale:
            X_train = self.scale_data(X_train)
            X_test = self.scale_data(X_test)

        y_train = pd.read_csv(data_folder_path+'/y_train'+sub_data_section+'.csv').to_numpy()
        y_test = pd.read_csv(data_folder_path+'/y_test'+sub_data_section+'.csv').to_numpy()
        
        y_train = y_train.reshape(-1,) # for sklearn 
        y_test = y_test.reshape(-1,)

        return X_train, X_test, y_train, y_test

    
    def scale_data(self,data):
        scale_obj = StandardScaler()
        return scale_obj.fit_transform(data)


    def remove_categoricals(self, data_in):
        data_out = data_in
        for col in data_in.columns:
            if 'Soil' in col or 'Wilderness' in col:
                data_out = data_out.drop(labels=[col], axis=1)
        return data_out