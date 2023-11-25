import pandas as pd
import urllib
import utils


def clean_data_captions(path):
    data = pd.read_csv(path, index_col=0)
    data['name'] = data['name'].map(lambda x: urllib.parse.quote(x, safe=":/"))
    return data


if __name__ == '__main__':
    parameters = utils.load_parameters('configs/preprocess.yaml')
    captions_parameters = parameters['data_captions']
    cleaned_data_captions = clean_data_captions(captions_parameters['in_path'])
    utils.safe_saving(cleaned_data_captions, captions_parameters['out_path'])
