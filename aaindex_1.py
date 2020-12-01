class AAindex_read_data():
    
    def __init__(self,file,filter_property_index=['GEIM800103','MUNV940101','OOBM770104','HUTJ700102','PALJ810115',
                                                  'MITS020101','QIAN880132','KARP850103','OOBM850102','FAUJ880113',
                                                  'NADH010106','ISOY800106','RADA880106','RACS820113','QIAN880112',
                                                  'GEOR030105','WEBA780101','QIAN880114','QIAN880125','DIGM050101',
                                                  'JOND750101','MIYS850101','QIAN880124']):
    
        self.file=file
        self.amino_symbol_order=    ['A','R','N','D','C','Q','E','G','H','I',
                     'L','K','M','F','P','S','T','W','Y','V','X']
        self.amino_symbol_dict= {0:'A',1:'R',2:'N',3:'D',4:'C',5:'Q',6:'E',7:'G',8:'H',9:'I',
                     10:'L',11:'K',12:'M',13:'F',14:'P',15:'S',16:'T',17:'W',18:'Y',19:'V',20:'X'}
        self.filter_property_index=  filter_property_index
        
    def get_primary_amino_data(self):
        
        with open(self.file,"r") as file:
            total_property=[]
            each_property=[]
            for line in file.readlines():
                line=line.strip("\n") 
                if line!="//":
                    each_property.append(line)
                else:
                    total_property.append(each_property)
                    each_property=[]
        
        return total_property
    
    def amino_property_filter_NA_processing(self,total_property):
        
        aminos_property={}
        NA_Accession_number=[]
        i=0
        for each in total_property:
            i+=1
            Accession_number=each[0].split(" ")[-1] 
            index_data=[]
            index=each[-2]+each[-1]
            index=index.split(" ") 
    
            for  number in index:
                if number=='NA':
                    NA_Accession_number.append(Accession_number)
                if number!='' and number!='NA':
                    index_data.append(float(number))

            aminos_property[Accession_number]=index_data
            index_data=[]

        NA_Accession_number=list(set(NA_Accession_number))
        def del_NA_property(aminos_property,NA_Accession_number):
            
            for Accession in NA_Accession_number:
                del aminos_property[Accession]
            
            return aminos_property

        amino_property=del_NA_property(aminos_property,NA_Accession_number)
    
        return amino_property
    
    def filter_property(self,amino_property):
        
        filter_property= []
        for amino_property_index in self.filter_property_index:
            filter_property.append(amino_property[amino_property_index])
        
        return filter_property
        
    
    def amino_transfer_property(self,pep_seq,amino_property):
        
        if self.filter_property_index:
            property_for_transfer= self.filter_property(amino_property)
        else:
            property_for_transfer= amino_property.values()
        amino_transfer_into_property=[]
        peptide_transfer_into_property= []
        for amino_acid in pep_seq:
            index_in_aminos_property=list(self.amino_symbol_dict.keys())[
                list(self.amino_symbol_dict.values()).index(amino_acid)]
            for amino_property in property_for_transfer:
                amino_transfer_into_property.append(amino_property[index_in_aminos_property])
            peptide_transfer_into_property.append(amino_transfer_into_property)
            amino_transfer_into_property=[]
        
        return peptide_transfer_into_property 
    
    def normalization(self,amino_property):
        
        from sklearn import preprocessing

        amino_property_normalization = preprocessing.normalize(amino_property,norm='l2')
        min_max_scaler = preprocessing.MinMaxScaler()
        amino_property_minmax = min_max_scaler.fit_transform(amino_property)
        
        return amino_property_normalization,amino_property_minmax
    
    def max_min_normal(self,amino_property):
        
        import numpy as np
        from sklearn import preprocessing
        
        new_amino_property = {}
        indexes = amino_property.keys()
        properties = amino_property.values()
        properties = list(properties)
        properties = np.array(properties)
        
        min_max_scaler = preprocessing.MinMaxScaler()
        
        aa_properties=[]
        for aa_property in properties:
            aa_property_dim = np.expand_dims(aa_property,axis=1)
            aa_property_dim = min_max_scaler.fit_transform(aa_property_dim)
            aa_property_dim = np.squeeze(aa_property_dim,axis=1)
            aa = []
            for i in aa_property_dim:
                aa.append(i)
            aa_properties.append(aa)
        for i,index in enumerate(indexes):
            new_amino_property[index] = aa_properties[i]
            
        return new_amino_property
    
    def max_normal(self,amino_property):
        
        import numpy as np
        from sklearn import preprocessing
        
        new_amino_property = {}
        indexes = amino_property.keys()
        properties = amino_property.values()
        properties = list(properties)
        properties = np.array(properties)
        
        aa_properties=[]
        for aa_property in properties:
            aa_property_dim = np.expand_dims(aa_property,axis=1)
            aa_property_dim_mean = aa_property_dim.mean(axis=0)
            aa_property_dim = aa_property_dim-aa_property_dim_mean
            aa_property_dim_std = aa_property_dim.std(axis=0)
            aa_property_dim = aa_property_dim/aa_property_dim_std
            aa_property_dim = np.squeeze(aa_property_dim,axis=1)
            aa = []
            for i in aa_property_dim:
                aa.append(i)
            aa_properties.append(aa)
        for i,index in enumerate(indexes):
            new_amino_property[index] = aa_properties[i]
            
        return new_amino_property
    
def peptide_into_property(peptides,filter_property_index,max_min=False):
    
    import numpy as np
    amino_property_1={}
    transfer_peptides=[]
    aaindex=    AAindex_read_data('aaindex_1.txt',filter_property_index)
    total_property= aaindex.get_primary_amino_data()
    amino_property= aaindex.amino_property_filter_NA_processing(total_property)
    if max_min:
        amino_property= aaindex.max_min_normal(amino_property)
    else:
        amino_property= aaindex.max_normal(amino_property)
    #amino_property_keys=amino_property.keys()
    #amino_property_values=np.array(list(amino_property.values()))
    #_,amino_property_values= aaindex.normalization(amino_property_values)
    #for number,index in enumerate(amino_property_keys):
     #   amino_property_1[index]=amino_property_values[number].tolist()
    for peptide in peptides:
        transfer_peptide= aaindex.amino_transfer_property(peptide,amino_property)
        transfer_peptides.append(transfer_peptide)
    transfer_peptides=  np.array(transfer_peptides)
    #transfer_peptides=  np.mean(transfer_peptides,axis=1)
    #filter_property=    aaindex.filter_property(amino_property)
    
    return transfer_peptides

import numpy as np
from sklearn import preprocessing
aaindex=    AAindex_read_data('aaindex_1.txt')
total_property= aaindex.get_primary_amino_data()
amino_property= aaindex.amino_property_filter_NA_processing(total_property)
properties =aaindex.max_normal(amino_property)