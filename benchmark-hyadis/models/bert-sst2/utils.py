import re

# reading given tsv file
with open("./data/test.tsv", 'r') as myfile:
    with open("./data/test.csv", 'w') as csv_file:
        for line in myfile:
            # Replace every tab with comma
            fileContent = re.sub("\t", ",", line)

            # Writing into csv file
            csv_file.write(fileContent)

# output
print("Successfully made csv file")