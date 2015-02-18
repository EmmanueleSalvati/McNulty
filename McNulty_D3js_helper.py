"""Functions to extract TSV files for D3 bar charts"""

from McNulty_feature_selection import retrieve_dataframe


if __name__ == '__main__':
    men = retrieve_dataframe()

    cp_men = men['diagnosis'].value_counts()
    with open('diag.tsv', 'w') as tsv:

        for ind in cp_men.index:
            tsv.write('%s\t%s\n' % (ind, cp_men[ind]))
