import requests 

URL="http://127.0.0.1:5050/predict"

FILE_PATH='jack_hammer.wav'

if __name__=="__main__":

    f=open(FILE_PATH,"rb")
    values={"file":(FILE_PATH,f,"audio/wav")}
    response=requests.post(URL,files=values)
    data=response.json()
    print("predicted keyword: {}".format(data["keyword"]))