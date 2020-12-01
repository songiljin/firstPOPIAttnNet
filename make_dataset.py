from aaindex_1 import peptide_into_property
from read_csv import MHC_Read_data,Processing_peptides,mhci_hla_a_data,mhci_data,mhcii_data
import config
from sklearn.model_selection import  train_test_split
import tensorflow as tf

    
class Dataset():
    def __init__(self):
    
        if config.mhc_type=='mhci':
    #all_peptide_seqs_mhci_hla_a,mhc_hla_a_label,positive_length_mhc_hla_a= mhci_hla_a_data()
            self.train_peptides,self.train_labels,self.positive_length,self.val_peptides,self.val_labels,self.test_peptides,self.test_labels=    mhci_data()
            #self.all_peptide_seqs,self.label,self.positive_length=    mhci_data()
        elif config.mhc_type=='mhcii':    
            self.train_peptides,self.train_labels,self.positive_length,self.val_peptides,self.val_labels,self.test_peptides,self.test_labels=    mhcii_data()
        elif config.mhc_type=='mhci_hla':
            #self.all_peptide_seqs,self.label,self.positive_length=    mhci_hla_a_data()
            self.train_peptides,self.train_labels,self.positive_length,self.val_peptides,self.val_labels,self.test_peptides,self.test_labels=    mhci_hla_a_data()
    
    def get_dataset(self,peptides,labels,positive_length):
        
        datasets= peptide_into_property(peptides,0,max_min=False)  #property_length=553
        c_datasets=datasets.astype('float32')
    
        return c_datasets[:2*positive_length],labels[:2*positive_length],positive_length
    

    
        
#dataset,label,positive_length=get_dataset(all_peptide_seqs,label,positive_length)


    def make_dataset(self,dataset,label):
    
        BUFFER_SIZE = len(dataset)
        BATCH_SIZE = config.BATCH_SIZE
        datasets = tf.data.Dataset.from_tensor_slices((dataset,label)).shuffle(BUFFER_SIZE)
        datasets = datasets.cache().batch(BATCH_SIZE, drop_remainder=True)
        print(1)
        
        return datasets

        #test_dataset=make_dataset(test_data,test_label)
    def train_val_test_dataset(self):

        train_data,train_label,positive_length = self.get_dataset(self.train_peptides,self.train_labels,self.positive_length)
        val_data =  peptide_into_property(self.val_peptides,0,max_min=False)
        val_data = val_data.astype('float32')
        val_label = self.val_labels
        test_data = peptide_into_property(self.test_peptides,0,max_min=False)
        test_data = test_data.astype('float32')
        test_label = self.test_labels
        train_dataset = self.make_dataset(train_data,train_label)
        val_dataset = self.make_dataset(val_data,val_label)
        test_dataset = self.make_dataset(test_data,test_label)
        
        return train_dataset,val_dataset,test_dataset
    
class_dataset=Dataset()
train_dataset,val_dataset,test_dataset=class_dataset.train_val_test_dataset()
train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels=    mhci_hla_a_data()