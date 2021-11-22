import defopt

from datasets import create_rls_genus_dataset, create_rls_species_dataset, create_test_dataset

if __name__ == '__main__':
    defopt.run([create_rls_genus_dataset, create_rls_species_dataset, create_test_dataset])
