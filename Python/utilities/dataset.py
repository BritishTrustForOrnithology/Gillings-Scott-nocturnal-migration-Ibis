# code primarily developed by Chris Scott with some modifications by Simon Gillings
# March 2020

import librosa
import numpy as np
import glob, os
import hashlib
import h5py
import pandas as pd
import pathlib
import pickle
import torch
from tqdm import tqdm_notebook as tqdm
import uuid

class Dataset(torch.utils.data.Dataset):
    r"""A child class of torch.utils.data.Dataset.

    __getitem__ fetches a data sample for a given key.
    __len__ returns the size of the dataset.

    """
    
    def __init__(self):
        self._df = pd.DataFrame()
        
        self._savename = 'dataset.pkl'
        self._savedir = 'tmp'
        if not os.path.isdir(self._savedir):
            os.mkdir(self._savedir)
            print('created tmp save directory')
        
        self._h5pyfile = os.path.join(self._savedir, self._savename).replace('.pkl', '.hdf5')
        
        self._sr = 22050
        self._duration = 2.0
        self._num_classes = 0
        self._valid_indices = []
        self._mapping = None
        self._columns = ['filename', 'offset', 'md5', 'id-type', 'outfile', 'target']
    
        
    def _save(self):
        outfile = os.path.join(self._savedir, self._savename)
        with open(outfile, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def _return_dictionary_from_df_row(self, row):
        file = row['filename']
        offset = row['offset']
        md5 = row['md5']
        id_type = row['id-type']
        return {'filename': file, 'offset': offset, 'md5': md5, 'id-type': id_type, 'outfile': 'NA', 'target': 0}

    def _save_clips(self, df, mode='w'):
        results = []

        with h5py.File(self._h5pyfile, mode) as hf:        
            print('\nloading clips...')
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                file = row['filename']
                offset = row['offset']

                try:
                    x, sr = librosa.load(file, sr=self._sr, mono=False, offset=offset-self._duration/2, duration=self._duration)
                except:
                    print('unable to load {}'.format(file))
                    continue

                #SG at this point may still be stereo file, so x.size is bigger, so divide by x.ndim so calc works
                #double // ensures the answer is int, for comparison to be robust
                # skip clip if too close to the edge of the recording
                if ((x.size // x.ndim) != int(self._sr*self._duration)):
                    print('Offset near edge - skipped file: {}, with offset: {}'.format(file, offset))
                    continue

                # extract the fields we want to save in the local dataframe
                d = self._return_dictionary_from_df_row(row)
                
                # this bit needs to be done if only a mono recording
                if (x.ndim == 1):
                    # create a unique filename
                    outfile = uuid.uuid4().hex
                    hf[outfile] = x
                    d['outfile'] = outfile
                    results.append(d)

                # this bit needs to be done for each channel but only if the file is stereo
                if (x.ndim == 2):
                    usech1 = row['ch1']
                    usech2 = row['ch2']
                    if (usech1 == 1):
                        # create a unique filename
                        outfile = uuid.uuid4().hex
                        hf[outfile] = x[0,:]
                        d['outfile'] = outfile
                        results.append(d)
                    if (usech2 == 1):
                        d = d.copy()
                        # create a unique filename
                        outfile = uuid.uuid4().hex
                        hf[outfile] = x[1,:]
                        d['outfile'] = outfile
                        results.append(d)
                        
        # save the valid clips to a local dataframe
        if (mode == 'w'):
            # write mode
            self._df = pd.DataFrame(results, columns=self._columns)
            self._valid_indices = self._df.index.tolist()
        elif (mode == 'a'):
            # append mode
            self._df = self._df.append(pd.DataFrame(results, columns=self._columns), ignore_index=True)
            self._valid_indices = self._df.index.tolist()
        else:
            # not implemented
            print('invalid mode: {}'.format(mode))
    
    def __len__(self):
        return len(self._valid_indices)
    
    def __getitem__(self, idx):
        # find dataframe index from valid_indices
        index = self._valid_indices[idx]
        with h5py.File(self._h5pyfile, 'r') as hf:
            outfile = self._df.iloc[index]['outfile']
            # x is the clip waveform
            x = hf.get(outfile)[:]
            
            # target is an integer
            target = self._df.iloc[index]['target'].astype('int32')
        return x, target

    # load an existing dataset
    def load(self):
        outfile = os.path.join(self._savedir, self._savename)
        if not pathlib.Path(outfile).exists():
            raise FileNotFoundError('no dataset .pkl file found to load')
        else:
            with open(outfile, 'rb') as f:
                tmp_dict = pickle.load(f)
                self.__dict__.update(tmp_dict)

    # compile a new dataset
    def compile(self, path: str, sr: int=22050, duration: float=2.0):
        df = self.labels_to_df(path)
        df = self.label_data_wrangling(df)
        
        # set the sampling rate and clip duration
        self._sr = sr
        self._duration = duration

        # save clips in write mode
        self._save_clips(df, mode='w')
        
        # default target
        self.map_target()
        
        # auto-save
        self._save()
      
 
    def map_target(self, column='id-type', mapping={'RE-F': 1}):
        self._mapping = mapping
        valid_columns = ['id-type', 'id', 'type']
        if column not in valid_columns:
            raise Exception("'{}' is not a valid column name, valid column names are: {}".format(column, valid_columns))
        
        self._df['target'] = 'NA'
        if (column == 'id'):
            self._df['target'] = self._df['id-type'].str.split('-', n=1, expand=True)[0].map(mapping).fillna(0)
        elif (column == 'type'):
            self._df['target'] = self._df['id-type'].str.split('-', n=1, expand=True)[1].map(mapping).fillna(0)
        else:
            self._df['target'] = self._df['id-type'].map(mapping).fillna(0)
        
        self._valid_indices = self._df.index.tolist()
            
        # remove clips with target==0 that overlap clips with target==1
        # clips are not deleted from the dataset, just removed from the current list of valid clips
        mask = np.ones_like(np.asarray(self._valid_indices))
        gb = self._df.groupby('filename')
        for group_name, df_group in gb:
            n_clips = len(df_group.index)
            if (n_clips > 1):
                df = df_group[['target', 'offset']].reset_index()
                offsets1 = df[df.target==1].offset.to_numpy()
                if (offsets1.size > 0):
                    offsets0 = df[df.target==0].offset.to_numpy()
                    indices = df[df.target==0]['index'].to_numpy()
                    for idx, offset in enumerate(offsets0):
                        closest_pos = np.abs(offsets1-offset).min()
                        # could use self._duration/2, but to be safe use self._duration
                        if (closest_pos < self._duration):
                            print('Target conflict - removing clip from {} with offset {}'.format(group_name, offset))
                            mask[indices[idx]] = 0
        self._valid_indices = np.asarray(self._valid_indices)[mask==1]
        
        # calculate number of classes
        self._num_classes = len(self._df.iloc[self._valid_indices].target.unique())
        
        # auto-save
        self._save()

        
    # subset the dataset into training and validation datasets
    # Stratified as it takes a fixed percentage of each class (target). Note that this can still be imbalanced
    def random_split_stratified_by_target(self, class_fraction: float=0.1) -> (torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset):
        # subset valid clips
        df = self._df.iloc[self._valid_indices].reset_index()
       
        gb = df.groupby('target', group_keys=False)
        validation_idx = gb.apply(lambda x: x.sample(frac=class_fraction)).index
        training_idx = df.drop(validation_idx, errors="ignore").index
        #print(validation_idx)

        # subset dataset
        validation_set = torch.utils.data.dataset.Subset(self, indices=validation_idx.to_list())
        training_set = torch.utils.data.dataset.Subset(self, indices=training_idx.to_list())
        return training_set, validation_set



    def _print_h5py_files(self):
        with h5py.File(self._h5pyfile, 'r') as f:
            print([key for key in f.keys()])
    
    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def sr(self) -> int:
        return self._sr

    @property
    def duration(self) -> float:
        return self._duration
    
 
    def label_data_wrangling(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'offset' not in df.columns:
            df['offset'] =  0.5 * (df['start'] + df['end'])

        #if using a lookup to merge codes together, simplify here
        #if lookup:    
        #    #lookup = {'GJ-F': 'XX-F', 'B_-F':'XX-F', 'ST-F':'XX-F'}
        #    df = df.replace(regex=lookup)    


        # default values
        df['id'] = df['label']
        df['type'] = 'NA'
        df['ch1'] = '1'
        df['ch2'] = '0'

        label_split = df['label'].str.split('-', n=2, expand=True)

        n_cols = len(label_split.columns)
        if n_cols == 2:
            df['id'] = label_split[0]
            df['type'] = label_split[1]
        if n_cols == 3:
            df['id'] = label_split[0]
            df['type'] = label_split[1]
            df['ch1'] = label_split[2].str[0]
            df['ch2'] = label_split[2].str[1]

        # if mixed labels need to replace None values
        df['type'].fillna('NA', inplace=True)
        df['ch1'].fillna('1', inplace=True)
        df['ch2'].fillna('0', inplace=True)

        #error checking
        df['error_label_length'] = df['label'].apply(lambda x: 0 if len(x) in (4,7) else 1)
        df['error_id_length'] = df['id'].apply(lambda x: 0 if len(x) == 2 else 1)
        df['error_ch1'] = df['ch1'].apply(lambda x: 0 if x in ('0', '1') else 1)
        df['error_ch2'] = df['ch2'].apply(lambda x: 0 if x in ('0', '1') else 1)
        df['error_type'] = df['type'].apply(lambda x: 0 if x in ('F', 'W', 'S', 'Z') else 1)
        df['errors'] = df['error_label_length'] + df['error_id_length'] + df['error_ch1'] + df['error_ch2'] + df['error_type']
        #export the errors
        errors = df[df['errors'] >= 1]
        if errors.shape[0] > 0:
            print('There are ' + str(errors.shape[0]) + ' labels with errors')
            print('Errors saved to csv')
            errors.to_csv('label_errors.csv', sep = ',', header=True, index=False)
        #keep just the OK rows
        df = df[df['errors'] == 0]
        df = df.drop(['errors', 'error_label_length', 'error_id_length', 'error_ch1', 'error_ch2', 'error_type'], axis=1)

        #type conversion    
        df['ch1'] = df['ch1'].astype('int32')
        df['ch2'] = df['ch2'].astype('int32')

        df['id-type'] = df['id'] + '-' + df['type']
        return df

    def labels_to_df(self, path: str, audacity_labels: bool=True, recurse: bool=True) -> pd.DataFrame:
        audio_exts = {'.flac', '.mp3', '.wav'}

        ext, sep, names = ('.txt', '\t', ['start', 'end', 'label'])
        if not audacity_labels: # then assume sonic visualiser labels
            ext, sep, names = ('.csv', ',', ['offset', 'label'])

        if recurse:
            all_files = [filename for filename in pathlib.Path(path).rglob('*' + ext)]
        else:
            all_files = [filename for filename in pathlib.Path(path).glob('*' + ext)]

        data = []
        for f in all_files:
            try:
                df = pd.read_csv(f, sep=sep, header=None, names=names, index_col=False)
            except:
                print('not a valid label file {}'.format(f))
                continue

            if (df.isnull().values.any()):
                print('NaN values found in {}'.format(f))
                continue

            # add md5 hash of label file
            df['md5'] = hashlib.md5(pathlib.Path(f).read_bytes()).hexdigest()

            # look for label filename but any extension
            file_list = glob.glob(os.path.splitext(f)[0] + '.*')

            # extract the extensions from the results
            ext_list = [os.path.splitext(file)[1] for file in file_list]

            # should only find a label file and an audio file
            if (len(ext_list)>2):
                print('multiple extensions found {}'.format(file_list))

            # search for a valid audio file ext in the ext_list
            audio_found = False
            for ext in ext_list:
                if ext.lower() in audio_exts:
                    # found a valid audio ext so add filename to dataframe
                    audio_found = True
                    df['filename'] = os.path.splitext(f)[0] + ext

            if not audio_found:
                print('no associated audio file found for {}'.format(f))
                continue
            
            data.append(df)


        df = pd.concat(data, ignore_index=True)
        return df

    def tabulate_targets(self):
        print('How big is the training dataset?')
        print(self._df[['filename','target']].groupby(['target']).agg('count'))
        
    #export a copy of the dataset as a csv to be able to assess sample sizes    
    def export_dataset(self):
        self._df.to_csv('dataset.txt', sep='\t', header = True, index = False)
    

        
print('loaded dataset module')