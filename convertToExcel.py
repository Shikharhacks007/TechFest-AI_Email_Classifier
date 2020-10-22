import os
import csv 

def cleaning( data ):
    '''
    Remove ' at front and back
    '''    

    data = data.decode('utf8')
    #data = data[2:-1]

    #print( "\n\nDATA:\n\n", data )
    
    clean_data = ""
    getRid = ["\n", "'", "\"", "."]
    for i in data:
        #i = str(i)
        if i not in getRid:
            clean_data += i
        else:
            clean_data += ' '

    #clean_data = clean_data.decode('utf8')

    return clean_data


csvfile = open( "newsgroupDocuments.csv", "a+" )
writer = csv.writer( csvfile )

upper = 100
filepath = ["business/business", "entertainment/entertainment", 
            "food/food", "graphics/graphics", "historical/historical", 
            "medical/medical", "politics/politics", "space/space",
            "sport/sport", "technologie/technologie" ]

label = ["Business", "Entertainment", "Food", "Graphics", "Historical", 
         "Medical", "Politics", "Space", "Sport", "Technology" ] 

for k in range(10):
    for file_number in range(1, upper+1):

        print("READ THIS" + f"./NewsgroupDocuments/{filepath[k]}_{file_number}.txt"  )

        f = open( f"./NewsgroupDocuments/{filepath[k]}_{file_number}.txt", 'rb' )
        
        '''
        dataStr = ""
        for line in f.readlines():
            try:
                dataStr += line
            except:
                print("Error!")
        '''

        data = f.read()
        #clean_data = cleaning( data )
        writer.writerow( [label[k], data] )

csvfile.close()

##svfile = open( "newsgroupDocuments.csv", "r+" )
##reader = csv.reader( csvfile, delimiter=',' )

f = open( "newsgroupDocuments.csv", "r" )
csvfile = open( "newsgroupDocumentsFinal.csv","a+" )
write = csv.writer( csvfile )

for k in f.readlines():
    
    label = ""
    for num in range( len(k) ):
        if temp != ',':
            label += temp
        else:
            break

    


print("Done")
