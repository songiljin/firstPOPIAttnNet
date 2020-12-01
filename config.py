BATCH_SIZE = 64
mhcip_units = 64
mhcip_units_2 = 64
mhcip_units_3 = 64
units= 512
w1_units= 64
peptide_length = 9
property_length = 553
fc_dim = 16
mhc_type='mhci_hla'
#mhc_type = 'mhcii'
#mhc_type='mhci'
if mhc_type=='mhcii':
    peptide_length=15
else:
    peptide_length=9
train_test_percentage = 0.15
lr = 0.001
EPOCHS= 30
input_shape=[peptide_length,property_length]
test_percent = 0.1
