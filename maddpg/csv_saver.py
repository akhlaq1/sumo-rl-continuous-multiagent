import csv


class CSV_Writer():
    def __init__(self, outfile="output"):
        self.header = []
        self.row = []
        self.outfile = outfile
        self.f = ''

    def create_file(self):
        self.f = open(self.outfile + '.csv', 'w', encoding='UTF8', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow(self.header)

    def write_row(self, data):
        self.writer.writerow(data)

    def close_file(self):
        self.f.close()