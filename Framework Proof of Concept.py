#Imports

#Data manipulation
import pandas as pd
import numpy as np
#Numeric machine learning
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential 
from tensorflow.keras.layers import LSTM,Dense, Input  
import tensorflow as tf
#Visualisation
import matplotlib.pyplot as plt
import networkx as nx
#text pre-processing
import nltk
from nltk.corpus import stopwords as nltkstopwords
from nltk.stem import WordNetLemmatizer
import string
import fitz  
#Text machine learning
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel

#Major variables
global timestep
timestep=1
global CSV_rows
CSV_rows=45632 
global test_rows
test_rows=26193 
global columns
columns=67

global CSV1
CSV1="CSV_1.csv"
global CSV2
CSV2="CSV_2.csv"

global epoch_num
epoch_num=1
global batch_num
batch_num=15
 
global topic_count
topic_count=4

global link_depth 
link_depth=50

global analysis_file
analysis_file="analysis.pdf"

global lda_topics
lda_topics=20
global lda_epoch
lda_epoch=25
global display_words
display_words=10

#Functions
def detection_datasets():
    #Read CSVs
    CSV=pd.read_csv(CSV1,on_bad_lines="skip") 
    test=pd.read_csv(CSV2,on_bad_lines="skip")
   
   
    #Remove columns that interfere/are uneeded with LSTM
    CSV_sort=CSV.drop(columns=["expiration_id","src_ip","src_mac","src_oui","dst_ip","dst_mac","dst_oui","ip_version","vlan_id","tunnel_id","application_category_name","application_is_guessed","requested_server_name","client_fingerprint","server_fingerprint","user_agent","content_type","application_name","Activity","Stage","DefenderResponse"])
    #Converts strings to binary value
    CSV_sort["Signature"]=CSV_sort["Signature"].replace("APT",1)
    #Fills blank
    CSV_sort["Signature"]=CSV_sort["Signature"].fillna(0)

    #Repeat of above on testing csv
    test_sort=test.drop(columns=["expiration_id","src_ip","src_mac","src_oui","dst_ip","dst_mac","dst_oui","ip_version","vlan_id","tunnel_id","application_category_name","application_is_guessed","requested_server_name","client_fingerprint","server_fingerprint","user_agent","content_type","application_name","Activity","Stage","DefenderResponse"])
    test_sort["Signature"]=test_sort["Signature"].replace("APT",1)
    test_sort["Signature"]=test_sort["Signature"].fillna(0) 
   
    #Debugging file export
    #CSV_sort.to_csv("CSV_Clean.csv",index=False)
    #CSV_sort.to_csv("Test_Clean.csv",index=False)

    return CSV_sort,test_sort


def detection_lstm(CSV,test):
   
    #Gets CSV minus the metric lstm is aimed at
    data=CSV.iloc[:,:-1]
    label=CSV.iloc[:,-1]
    test_data=test.iloc[:,:-1]

   
   
    #Normalises data
    scale=MinMaxScaler(feature_range=(0,1))

    #Transforms data
    data_scale=scale.fit_transform(data)   
    test_scale=scale.fit_transform(test_data) 

   
    #Scales data to right format
    data_scale=data_scale.reshape(CSV_rows-1,timestep,columns)
    test_scale=test_scale.reshape(test_rows-1,timestep,columns) 

    
    #Splits data into learning and testing
    data_train,data_valid,label_train,label_valid=train_test_split(data_scale,label,test_size=0.15,random_state=1)

    #LSTM model
    lstm=Sequential()
    #Input layer
    lstm.add(Input(shape=(1,data_train.shape[2])))  
    #LSTM layer
    lstm.add(LSTM(100,dropout=0.1, return_sequences=False))
    #Dense layer
    lstm.add(Dense(1,activation="sigmoid"))

    #Compiles LSTM, binary crossentropy because target is binary choice
    lstm.compile(optimizer="adam",loss="binary_crossentropy")
    #Assign LSTM variables
    lstm.fit(data_train,label_train,epochs=epoch_num,batch_size=batch_num,verbose=0)  

    #Evaluate with split data
    lstm.evaluate(data_valid,label_valid) 
    #Predict values using testing data
    guess=lstm.predict(test_scale)
    #Converts numeric guess to binary
    guess=(guess>0.5) 

    #Save guess to base csv, base csv used to retain columns initially dropped for LSTM
    base_csv=pd.read_csv(CSV2,on_bad_lines="skip")
    base_csv["Guess"]=guess.flatten()
    base_csv["Signature"]=base_csv["Signature"].fillna("na")
    base_csv.to_csv("Detection_Output.csv",index=False) 

def detection_evaluation():
    #Calculates percentage accuracy of LSTM
    data=pd.read_csv(r"Detection_Output.csv",on_bad_lines="skip")
    counter=0

    #Overall accuracy
    for i,record in data.iterrows():
        if record["Signature"]=="APT" and record["Guess"]==True:
            counter=counter+1
        elif record["Signature"]=="na" and record["Guess"]==False:
            counter=counter+1
        
    percent=(counter/(test_rows-1))*100
    print("#############################################################")
    print("Overall accuracy:",percent,"%")   

    #Just APT accuracy
    apt_total=0
    counter=0
    for i,record in data.iterrows():
        if record["Signature"]=="APT" and record["Guess"]==True:
            apt_total=apt_total+1
            counter=counter+1
        elif record["Signature"]=="APT" and record["Guess"]==False:
            apt_total=apt_total+1
    apt_percent=(counter/apt_total)*100
    print("APT accuracy",apt_percent,"%")

    #APT type accuracy
    cover_count=0
    cover_total=0
    for i,record in data.iterrows():
        if record["Signature"]=="APT" and record["Guess"]==True and record["Stage"]=="Cover up":
            cover_total=cover_total+1
            cover_count=cover_count+1
        elif record["Signature"]=="APT" and record["Guess"]==False and record["Stage"]=="Cover up":
            cover_total=cover_total+1
    cover_percent=(cover_count/cover_total)*100
    print("Stage: cover up accuracy",cover_percent,"%")
    print(cover_count,cover_total)

    foothold_count=0
    foothold_total=0
    for i,record in data.iterrows():
        if record["Signature"]=="APT" and record["Guess"]==True and record["Stage"]=="Establish Foothold":
            foothold_total=foothold_total+1
            foothold_count=foothold_count+1
        elif record["Signature"]=="APT" and record["Guess"]==False and record["Stage"]=="Establish Foothold":
            foothold_total=foothold_total+1
    foothold_percent=(foothold_count/foothold_total)*100
    print("Stage: establish foothold accuracy",foothold_percent,"%")
    print(foothold_count,foothold_total)

    exfil_count=0
    exfil_total=0
    for i,record in data.iterrows():
        if record["Signature"]=="APT" and record["Guess"]==True and record["Stage"]=="Data Exfiltration":
            exfil_total=exfil_total+1
            exfil_count=exfil_count+1
        elif record["Signature"]=="APT" and record["Guess"]==False and record["Stage"]=="Data Exfiltration":
            exfil_total=exfil_total+1
    exfil_percent=(exfil_count/exfil_total)*100
    print("Stage: data exfiltration accuracy",exfil_percent,"%")
    print(exfil_count,exfil_total)
   

def detection_graphing():
    #2nd CSV used to match with csv used in final lstm output.
    graph_csv=pd.read_csv(CSV2,on_bad_lines="skip")
    #Flowrate
    sd_flow=[]
    ds_flow=[]

    #Gets values from relevant columns
    for i,record in graph_csv.iterrows():
        sd_value=record["src2dst_bytes"]
        sd_flow.append(sd_value)
        ds_value=record["dst2src_bytes"]
        ds_flow.append(ds_value)

    #Gets the amount of entries to act as time, dataset is sequential
    time=range(len(sd_flow))
  
    #Plots and saves source-destination graph
    plt.plot(time,sd_flow,color="blue",label="sd")
    plt.title("Source-Destination Flow")
    plt.xlabel("Time (Sequential)")
    plt.ylabel("Flow (Bytes)")
    #plt.show()
    plt.savefig("sd_flow.png")
    plt.clf() 

    #Plots and saves destination-source graph
    plt.plot(time,ds_flow,color="red",label="ds")
    plt.title("Destination-Source Flow")
    plt.xlabel("Time (Sequential)")
    plt.ylabel("Flow (Bytes)")
    #plt.show()
    plt.savefig("ds_flow.png")
    plt.clf()
     
    #IP map
    ip_dict={}    
    
    for i,record in graph_csv.iterrows():
        #Gets IPs per record
        source_ip=str(record["src_ip"])
        destination_ip=str(record["dst_ip"])

        #If combo exists in dictionary increment otherwise add
        try:
         counter=ip_dict[(source_ip,destination_ip)]
         counter=counter+1
         ip_dict[(source_ip,destination_ip)]=counter
        except:
         ip_dict[(source_ip,destination_ip)]=1

    #Cuts entries with less frequency for cleaner output
    for keys,value in list(ip_dict.items()):
        if value <=link_depth:
         del ip_dict[keys]

    #Show Graphs
    link_graph=nx.DiGraph()
    ip_list=[]
    for keys,value in ip_dict.items():
        source=keys[0]
        destination=keys[1]

        if source not in ip_list:
            ip_list.append(source)

        if destination not in ip_list:
            ip_list.append(destination)

    for ip in ip_list:
        link_graph.add_node(ip)

    for keys,value in ip_dict.items():
        source=keys[0]
        destination=keys[1]
        link_graph.add_edge(source,destination)
    
    layout=nx.spring_layout(link_graph,k=1)
    nx.draw(link_graph,pos=layout,with_labels=True,font_size=10,node_size=10) 
    #plt.show()
    plt.savefig("link_graph.png")
    plt.clf()

    print("#############################################################")
    print("Detection graphs exported to file location")
    print("toggle the code for graphs in ui")    
    
 
def mitigation_rule():
    #True= All lstm guesses, False=Every labelled APT
    mode=False

    #Input data that will cause records to be flagged.  
    ips=["10.8.10.84"]
    ports=[3306]
    agents=["Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"]
    transfer_threshold=20000000

    print("#############################################################")
    print("Mitigation rules looking for:")
    print("IPS:")
    for ip in ips:
        print(ip)
    print("Ports:")
    for port in ports:
        print(port)
    print("User Agents:")
    for agent in agents:
        print(agent)
    print("Data transfer of ",transfer_threshold,"bytes")

    print("Detection guess only mode is currently:",mode)
    

    data=pd.read_csv(r"Detection_Output.csv",on_bad_lines="skip")
    #Add columns
    data[["IP_flag","Port_flag","Agent_flag","Transfer_flag","Violations"]]=None

    #Adjusts for modes
    if mode==True:
        data=data[data["Guess"]==True]
    elif mode==False:
        data=data[data["Signature"]=="APT"] 
        
    #IPs 
    target_ids=[]
    target_data=data[(data["src_ip"].isin(ips)) | (data["dst_ip"].isin(ips))]
    for i,record in target_data.iterrows():
        row_id=record["id"]
        target_ids.append(row_id)
    for value in target_ids:
        data.loc[data["id"]==value,"IP_flag"]=True
    
    #Ports
    target_ids=[]
    target_data=data[(data["src_port"].isin(ports)) | (data["dst_port"].isin(ports))]
    for i,record in target_data.iterrows():
        row_id=record["id"]
        target_ids.append(row_id)  
    for value in target_ids:
        data.loc[data["id"]==value,"Port_flag"]=True
   
    #Agents
    target_ids=[]
    target_data=data[data["user_agent"].isin(agents)]
    for i,record in target_data.iterrows():
        row_id=record["id"]
        target_ids.append(row_id)  
    for value in target_ids:
        data.loc[data["id"]==value,"Agent_flag"]=True
   
    #Threshold
    target_ids=[]
    target_data=data[(data["bidirectional_bytes"]>transfer_threshold) | (data["src2dst_bytes"]>transfer_threshold) | (data["dst2src_bytes"]>transfer_threshold)]
    for i,record in target_data.iterrows():
        row_id=record["id"]
        target_ids.append(row_id)   
    for value in target_ids:
        data.loc[data["id"]==value,"Transfer_flag"]=True 
  
   #Keeps count of number of flags per record and adds to CSV
    violation_list=[]
    for i,record in data.iterrows():
        violations=0
        if record["IP_flag"]==True:
            violations=violations+1
        if record["Port_flag"]==True:
            violations=violations+1 
        if record["Agent_flag"]==True:
            violations=violations+1
        if record["Transfer_flag"]==True:
            violations=violations+1
        violation_list.append(violations)

    data["Violations"]=violation_list
    
    data.to_csv("Mitigation_Output_Dataset.csv",index=False) 

def analysis_data():
    #Fitz lets pdf be opened
    open_text=fitz.open(analysis_file)
    #Blank data used as base
    text_data=""
    #Pages of pdf added to data as text
    for page in open_text:
        page_data=page.get_text()
        text_data=text_data+page_data
  
    #Pre-processing

    #Removes capitals
    text_data=text_data.lower()
    #Tokenises data
    text_data=nltk.word_tokenize(text_data)
    #Removes punctuation
    text_data=[text for text in text_data if text not in string.punctuation]

    #Removes common gap words
    stopwords=set(nltkstopwords.words("english"))
    text_data=[text for text in text_data if text not in stopwords]

    #Reverts words to basic forms 
    lemmatizer=WordNetLemmatizer()
    text_data=[lemmatizer.lemmatize(text) for text in text_data]

    #Additional tokenisation to work for Gensim
    #LDA used to run off of sklearn where this was uneeded
    text_data=[nltk.word_tokenize(text) for text in text_data]

    return text_data

def analysis_ml(text_data):
    #Splits text data as needed by LDA
    id2word=corpora.Dictionary(text_data) 
    lem_text=text_data

    #Create corpus
    corpus=[id2word.doc2bow(text) for text in lem_text]

    #Dictionary for graph
    umass_dict={}
    #List to get results
    umass_list=[]


    for i in range(1,lda_topics+1):

     #LDA model
     print("Iteration",i)
     lda=gensim.models.LdaModel(corpus=corpus,id2word=id2word,num_topics=i,per_word_topics=True,passes=lda_epoch,random_state=1) 

    #Evaluate model performance
     umass_model=CoherenceModel(model=lda,corpus=corpus, coherence="u_mass")
     umass_score=umass_model.get_coherence()
     umass_dict[i]=umass_score
     umass_list.append(umass_score) 
     
    #Gets best umass and index so output can be obtained
    best_umass=min(umass_list)
    best_index=umass_list.index(best_umass)
    best_index=best_index+1

    #Gets medium umass and index for results
    umass_list.sort()
    mid_index=len(umass_list)//2
    mid_umass=umass_list[mid_index]

    #Graph to display umass - topics, intended for large topic number tests
    graph_scores=list(umass_dict.values())
    graph_topics=list(umass_dict.keys())
    plt.scatter(graph_topics,graph_scores)
    plt.title("Umass - Topics")
    plt.xlabel("Topics")  
    plt.ylabel("Umass")
    plt.show()
    

    #Run LDA with best results
    lda=gensim.models.LdaModel(corpus=corpus,id2word=id2word,num_topics=best_index,per_word_topics=True,passes=lda_epoch,random_state=1) 
    print("########")
    print("best umass", best_umass)
    print("best index", best_index)
    print("LDA Topics:")
    print(lda.print_topics(num_words=display_words))

    #Run LDA with mid results
    lda=gensim.models.LdaModel(corpus=corpus,id2word=id2word,num_topics=mid_index,per_word_topics=True,passes=lda_epoch,random_state=1) 
    print("########")
    print("mid umass", mid_umass)
    print("mid index", mid_index)
    print("LDA Topics:")
    print(lda.print_topics(num_words=display_words))  
    print("########") 

      
def detection():
    CSV,test=detection_datasets()
    detection_lstm(CSV,test)   
    detection_evaluation()
    detection_graphing()
    

def mitigation():
    mitigation_rule()
   

def analysis():
    data=analysis_data()
    analysis_ml(data)

def setup():
 #Checks for and downloads needed data for pre-processing
 print("#############################################################")
 print("Checking for and downloading information needed")
 nltk.download("punkt_tab")
 nltk.download("stopwords")
 nltk.download("wordnet") 
 print("#############################################################")
    
def main():
    setup()
    detection()
    mitigation()
    analysis()




main()     