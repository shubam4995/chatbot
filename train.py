import tensorflow
import nltk
import colorama
import numpy
import flask
from sklearn.preprocessing import LabelEncoder

from keras.layers import Activation, Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data={"intents": [
{"tag": "greeting",
"patterns": ["Hi","Hey","Is anyone there?","Hello","Hay"],
"responses": ["Hello","Hi","Hi there"]
},
{"tag": "goodbye",
"patterns": ["Bye","See you later","Goodbye"],
"responses": ["See you later","Have a nice day","Bye! Come back again"]
},
{"tag": "offer",
"patterns": [" accepted", "offer"," online"," What do I do next","What"," next"," after"," getting"," offer","after offer","Next process ","offer"],
"responses": ["You need to complete the onboarding paperwork that you should see in an email request from Success Factors within 24 hours. This will include your background check consent form state and federal tax forms etc. Please ask for help immediately if you are having trouble completing these things."]
},
{"tag": "Location",
"patterns": ["assigned ","office","Location","my office location","office location"],
"responses": ["you will tagged to the closest office"]
},
{"tag": "Onboarding",
"patterns": ["Any contact details","reach out","onboarding","onoarding process"],
"responses": ["You will be receiving a calendar invite within 10 days of your start date for a New Hire Onboarding Call.  This will be a video call using Microsoft Teams   and will include all of the other people who will be starting with you on your first day.  You will have a chance to ask question meet and hear from your fellow new hires as well a receive instruction on handling the corporate equipment (laptop, logins, ID’s, etc) being sent to you shortly."]
},
{"tag": "Project",
"patterns": ["client project","assigned","project","type of work","project information"],
"responses": ["Once you start, you will coordinate with you supervising manager as to which team you will join initially.  This is a full-time, salaried role with Capgemini   and NOT a contract assignment with a single project.  Your career will hopefully be a very long, very engaging and very educational technical journey   across multiple teams and clients.  Where you start is simply where you will have the best opportunity to grow and learn proper software development   methodology before moving up within Capgemini."]
},
{"tag": "relocation",
"patterns": ["relocate","relocate from my place","relocation is neccessity"],
"responses": ["Due to COVID, we are all working remotely for the next several months – you can either work from home or go into any Capgemini office that is convenient to your current location.  As things begin to open back up in 2022, you may have the opportunity to relocate to a client site if you and your manager deem this as the most appropriate path for you.  We have clients everywhere across the United States and we want you to be flexible with relocation and your ability to entertain any and all locations. You will work with your team to determine the best account and location. Be open to all options."]
},
{"tag": "paycheck salary",
"patterns": ["paycheck","salary","date for salary"],
"responses": ["Capgemini paydays are on the 15th and last business day of the month."]
},
{"tag": "orientation",
"patterns": ["orientation ","orientation place","visit office for orientation"],
"responses": ["NO, due to COVID, all new hire orientation is now completed virtual and remote."]
},
{"tag": "Bench",
"patterns": ["Bench policy","bench period","bench duration","bench"],
"responses": ["There is no set policy for an employee to stay on the bench. Capgemini has a dedicated team of professionals to map & re-allocate internal bench employees to new projects. "]
},
{"tag": "reschedule background check",
"patterns": ["background check","rescheduled","schedule","Missed"],
"responses": ["You can contact your recruiter & they will be able to connect you with the background check team who can help in rescheduling."]
},
{"tag": "drug test",
"patterns": ["drug tested","Capgemini","client","drug test","test"],
"responses": ["You will not be drug tested on behalf of Capgemini, but the client may require it depending on what their onboarding process is like."]
},
{"tag": "H1",
"patterns": ["H1 B","documents","copy","H1"],
"responses": ["Write to Neha Pinjarkar <neha.pinjarkar@capgemini.com > requesting it from her with your Project Code. Your Project Code can be obtained from your reporting Manager. "]
},
{"tag": "allowance",
"patterns": ["relocation","allowance","relocation allowance"],
"responses": ["Yes we provided relocation allowance of $5000 ($3500 relocation + $1500 Lease breakage)."]
},
{"tag": "laptop",
"patterns": ["laptop","when","shipped","type"],
"responses": ["You will be getting an HP laptop. The laptop will be shipped a few days before your start date."]
},
{"tag": "travel",
"patterns": ["travel","breaking","lease","expenses","travel expenses","breaking a lease"],
"responses": ["All travel expenses are covered. You will get a relocation reimbursement up to $1500. Capgemini will break an active lease. "]
},
{"tag": "work",
"patterns": ["working","Capgemini","work in capgemini","choosing","Capgemini"],
"responses": ["We are the global leader in partnering with companies to transform and manage their business by harnessing the power of technologies. Capgemini is trusted by its clients to address the entire breadth of their business needs from strategy and design to operations. Capgemini reported 2020 global revenues of €16 billion."]
},
{"tag": "benefits",
"patterns": ["benefits","provided","capgemini","work benefits"],
"responses": ["Your recruiter will share your general benefits to you directly. We will also share a benefits summary document for your review. "]
},
{"tag": "stock",
"patterns": ["stock","purchase","capgemini stock plan","stock purchase"],
"responses": ["Yes, employees can purchase stock in Capgemini twice a year at the employee discounted price. "]
},
{"tag": "background ",
"patterns": ["background","check","not cleared","background check","not"," cleared","process"],
"responses": ["Yes, however, you will have to go through another background check. Anything client specific can be waived. "]
},
{"tag": "H1b",
"patterns": ["H1b" ,"sponsorship","sponsorship for H1b","H1b sponsorship" ,"capgemini"],
"responses": ["All candidates (valid EAD holders) that are employed should be qualified for H1b Sponsorship."]
},
{"tag": "US_benefits",
"patterns": ["US"," benefits","US benefits","Can i get US benefits"],
"responses": ["To enroll in benefits, please call our enrollment partner at 877-279-3639 between 8 am - 5 pm CST, Monday through Friday within 30 days of your hire date or your transfer date to the US."]
},
{"tag": "401K",
"patterns": ["401K","match"," percentage"],
"responses": ["Total maximum company match is 4%"]
},
{"tag": "GC",
"patterns": ["GC"," processing","GC processing","terms"],
"responses": ["Capgemini will do GC processing after 12 months of completion of employment as per our policy."]
},
{"tag": "I797",
"patterns": ["I797"," Approval"," Notice","I-9 Form","I797 Approval"],
"responses": ["The hard copy of I797 Approval Notice will be shipped to you by the Capgemini immigration team. Soft copies cannot be used to fill I-9 form because the e-verifier will need the hardcopy in-hand to verify for I9 step 2. New Joiners who have not received the Capgemini I797 approval hard copy can use their current petition with their current employer to fill the I-9 form. "]
},
{"tag": " benefits",
"patterns": ["furniture"," movers","relocation allowance benefits","benefits","new location","covers"],
"responses": ["No, the relocation allowance only covers the movers expense for your household & car. Please refer to the benefits document for details."]
},
{"tag": " ",
"patterns": [""],
"responses": ["Sorry"]
}

]}


#data= json.loads(open("intents.json").read())


training_sentences = []
training_labels=[]
labels = []
responses = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        training_sentences.append(pattern)
        training_labels.append(intent["tag"])
    responses.append(intent["responses"])

    if intent['tag'] not in labels:
        labels.append(intent["tag"])

num_classes = len(labels)



lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<00V>"

tokenizer = Tokenizer(num_words = vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_squences = pad_sequences(sequences, truncating="post", maxlen= max_len)


model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])


model.summary()


epochs = 500
his = model.fit(padded_squences, np.array(training_labels), epochs = epochs)

model.save("chat_model",his)



import pickle

# to save the fitted tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# to save the fitted label encoder
with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
