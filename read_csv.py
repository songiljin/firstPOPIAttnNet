import pandas as pd
import config

class MHC_Read_data():
    
    def __init__(self,file):
        
        self.total_data=pd.read_csv(file)
        self.mhci_data=self.total_data.loc[lambda df:
                              df.MHCType=='MHC-I']
        self.human_mhci_data=self.mhci_data.loc[lambda df:
                                    df.DatasetType=='Human']
        self.mhcii_data=self.total_data.loc[lambda df:
                              df.MHCType=='MHC-II']
        self.human_mhcii_data=self.mhcii_data.loc[lambda df:
                                    df.DatasetType=='Human']
    
    def human_mhci_subclass(self,HLA_Subtype):
        
        human_mhci_sub_positive=self.human_mhci_data.loc[lambda df:
                                        df.Immunogenicity=='Positive']
        human_mhci_sub_positive=human_mhci_sub_positive.fillna('nan', axis = 0)
        human_mhci_sub_positive=human_mhci_sub_positive[human_mhci_sub_positive
                            ['MHC_Restriction'].str.contains(HLA_Subtype)]
        human_mhci_sub_positive=human_mhci_sub_positive.loc[
            lambda df:df.Immunogenicity_Evidence!='nan']
        human_mhci_sub_negative=self.human_mhci_data.loc[
            lambda df:df.Immunogenicity=='Negative']
    
        return human_mhci_sub_positive,human_mhci_sub_negative
    
    def human_mhci(self):
        
        human_mhci_positive=self.human_mhci_data.loc[lambda df:
                                        df.Immunogenicity=='Positive']

        human_mhci_negative=self.human_mhci_data.loc[lambda df:
                                        df.Immunogenicity=='Negative']
        
        return human_mhci_positive,human_mhci_negative
        
    def human_mhcii(self):
        
        human_mhcii_positive=self.human_mhcii_data.loc[lambda df:
                                        df.Immunogenicity=='Positive']
        human_mhcii_positive=human_mhcii_positive.fillna('nan', axis = 0)
        human_mhcii_positive=human_mhcii_positive.loc[lambda df:
                                 df.Immunogenicity_Evidence!='nan']
        human_mhcii_negative=self.human_mhcii_data.loc[lambda df:
                                        df.Immunogenicity=='Negative']
        
        return human_mhcii_positive,human_mhcii_negative
    
    def get_sequences(self,positive,negative):
         
         positive_peptide_seqs=[]
         negative_peptide_seqs=[]
         for pos_peptide in positive['Peptide']:
             positive_peptide_seqs.append(pos_peptide)
        
         for neg_peptide in negative['Peptide']:
             negative_peptide_seqs.append(neg_peptide)
        
         return positive_peptide_seqs,negative_peptide_seqs
        
read_data=MHC_Read_data('mhc.csv')
human_mhci_hla_a_positive,human_mhci_hla_a_negative=    read_data.human_mhci_subclass('HLA-A')
human_mhci_hla_a_positive_seqs,human_mhci_hla_a_negative_seqs=    read_data.get_sequences(
    human_mhci_hla_a_positive,human_mhci_hla_a_negative)
human_mhci_positive,human_mhci_negative=  read_data.human_mhci()     
human_mhci_positive_seqs,human_mhci_negative_seqs=  read_data.get_sequences(
    human_mhci_positive,human_mhci_negative)   
human_mhcii_positive,human_mhcii_negative=  read_data.human_mhcii()
human_mhcii_positive_seqs,human_mhcii_negative_seqs=  read_data.get_sequences(
    human_mhcii_positive,human_mhcii_negative)   


class Processing_peptides():
    
    
    def length_filter(self,peptide_length,peptide_seqs):
        
        length_peptides=[]
        for peptide in peptide_seqs:
            if len(peptide)==peptide_length:
                #peptide_r=peptide[::-1]
                length_peptides.append(peptide)
        return  length_peptides
    
    def make_peptide_label(self,peptide_lengths,positive_seqs,negative_seqs):
        
        if peptide_lengths:
            positive_seqs=self.length_filter(peptide_lengths,positive_seqs)
            negative_seqs=self.length_filter(peptide_lengths,negative_seqs)
        positive_length=len(positive_seqs)
        label=[]
        for i in range(len(positive_seqs)+len(negative_seqs)):
            if (i<len(positive_seqs)):
                label.append(1)
            else:
                label.append(0)
        import numpy as np
        import random
        
        random.shuffle(positive_seqs)
        random.shuffle(negative_seqs)
        
        all_peptide_seqs=positive_seqs+negative_seqs
        
        return all_peptide_seqs,label,positive_length
    
   
    def seq_padding(self,pep_seq):
       
        pad_seq=[]
        for peptide in pep_seq:
            padding_length=30-len(peptide)
            if padding_length:
                pad_peptide=(padding_length//2)*'X'+peptide+(
                    padding_length-(padding_length//2))*'X'
                pad_seq.append(pad_peptide)
            else:
                pad_seq.append(peptide)
        return   pad_seq  
        
processing_peptides=Processing_peptides()       
all_peptide_seqs_mhci_hla_a,mhc_hla_a_label,positive_length_mhc_hla_a=    processing_peptides.make_peptide_label(
    9,human_mhci_hla_a_positive_seqs,human_mhci_hla_a_negative_seqs)

def mhci_hla_a_data():
    
    read_data=MHC_Read_data('mhc.csv')
    human_mhci_hla_a_positive,human_mhci_hla_a_negative=    read_data.human_mhci_subclass('HLA-A')
    human_mhci_hla_a_positive_seqs,human_mhci_hla_a_negative_seqs=    read_data.get_sequences(
        human_mhci_hla_a_positive,human_mhci_hla_a_negative)  
    processing_peptides=Processing_peptides()       
    all_peptide_seqs_mhci_hla_a,mhc_hla_a_label,positive_length=    processing_peptides.make_peptide_label(
        9,human_mhci_hla_a_positive_seqs,human_mhci_hla_a_negative_seqs)
    
    def make_train_test(peptides,labels,positive_length,test_percent):
        
        number = positive_length*test_percent
        number = int(number)
        train_peptides = peptides[number:-number]
        train_labels = labels[number:-number]
        test_peptides = peptides[:number]
        test_peptides_n = peptides[-number:]
        test_peptides.extend(test_peptides_n)
        test_labels = labels[:number]
        test_labels_n = labels[-number:]
        test_labels.extend(test_labels_n)
        positive_length = positive_length-number
    
        return train_peptides,train_labels,positive_length,test_peptides,test_labels
    train_peptides,train_labels,positive_length,test_peptides,test_labels= make_train_test(
        all_peptide_seqs_mhci_hla_a,mhc_hla_a_label,positive_length,0.15)
    
    train_peptides,train_labels,positive_length,val_peptides,val_labels= make_train_test(
        train_peptides,train_labels,positive_length,0.15)
    
    return train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels
    
def mhci_data():
    
    read_data=MHC_Read_data('mhc.csv')
    human_mhci_positive,human_mhci_negative=  read_data.human_mhci()     
    human_mhci_positive_seqs,human_mhci_negative_seqs=  read_data.get_sequences(
        human_mhci_positive,human_mhci_negative)   
    processing_peptides=Processing_peptides()       
    all_peptide_seqs_mhci,mhci_label,positive_length=    processing_peptides.make_peptide_label(
        9,human_mhci_positive_seqs,human_mhci_negative_seqs)
    def make_train_test(peptides,labels,positive_length,test_percent):
        
        number = positive_length*test_percent
        number = int(number)
        train_peptides = peptides[number:-number]
        train_labels = labels[number:-number]
        test_peptides = peptides[:number]
        test_peptides_n = peptides[-number:]
        test_peptides.extend(test_peptides_n)
        test_labels = labels[:number]
        test_labels_n = labels[-number:]
        test_labels.extend(test_labels_n)
        positive_length = positive_length-number
    
        return train_peptides,train_labels,positive_length,test_peptides,test_labels
    train_peptides,train_labels,positive_length,test_peptides,test_labels= make_train_test(
        all_peptide_seqs_mhci,mhci_label,positive_length,0.15)
    
    train_peptides,train_labels,positive_length,val_peptides,val_labels= make_train_test(
        train_peptides,train_labels,positive_length,0.15)
    
    return train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels
    
train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels=    mhci_data()

def mhcii_data():
    
    read_data=MHC_Read_data('mhc.csv')
    human_mhcii_positive,human_mhcii_negative=  read_data.human_mhcii()     
    human_mhcii_positive_seqs,human_mhcii_negative_seqs=  read_data.get_sequences(
        human_mhcii_positive,human_mhcii_negative)   
    processing_peptides=Processing_peptides()       
    all_peptide_seqs_mhcii,mhcii_label,positive_length=    processing_peptides.make_peptide_label(
        15,human_mhcii_positive_seqs,human_mhcii_negative_seqs)
    
    def make_train_test(peptides,labels,positive_length,test_percent):
        
        number = positive_length*test_percent
        number = int(number)
        train_peptides = peptides[number:-number]
        train_labels = labels[number:-number]
        test_peptides = peptides[:number]
        test_peptides_n = peptides[-number:]
        test_peptides.extend(test_peptides_n)
        test_labels = labels[:number]
        test_labels_n = labels[-number:]
        test_labels.extend(test_labels_n)
        positive_length = positive_length-number
    
        return train_peptides,train_labels,positive_length,test_peptides,test_labels
    train_peptides,train_labels,positive_length,test_peptides,test_labels= make_train_test(
        all_peptide_seqs_mhcii,mhcii_label,positive_length,0.15)
    
    train_peptides,train_labels,positive_length,val_peptides,val_labels= make_train_test(
        train_peptides,train_labels,positive_length,0.15)
    
    return train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels

train_peptides,train_labels,positive_length,val_peptides,val_labels,test_peptides,test_labels=    mhcii_data()