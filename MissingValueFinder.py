# If you have better solution feel free to make Pull Request  !

from numpy import *
# remember row in excel is column in real life
# Just change the row name to find whether in a particular row has empty elements or not . Eg row[11] 
import csv
 
inp = open('D:\Python\Heart Disease\original.csv','r')
out = open('D:\Python\Heart Disease\original1.csv','w') 

writer = csv.writer(out)

for row in csv.reader(inp):
	if(row[11] >= "0" and row[12] >= "0"):
		writer.writerow(row)
print(out)
inp.close()
out.close()





	
		
	