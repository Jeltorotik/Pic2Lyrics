import telebot
import main
import os.path

with open('bot_token.txt', 'r') as tkn:
    TOKEN = tkn.read()
    tkn.close()
        
bot = telebot.TeleBot(TOKEN)

def doc_type(document):
    return str(document.file_name.split('.')[-1])

def generate_labels(message_chat_id):
    pipeline = main.Pic2Lyrics()
    pipeline.video_2_images('data/{}.mp4'.format(message_chat_id))
    pipeline.image_2_labels()
    return pipeline.images[0][2] #TODO: Fix labels 

def generate_song(labels):
    pass # TODO: Сделать генератор по лейблам



@bot.message_handler(content_types=['video', 'photo', 'document'])
def mainfunc(message):

    #Downloading file
    if message.video is not None:
        print(message.video)
        file_info = bot.get_file(message.video.file_id)
    elif message.photo is not None:
        print(message.photo)
        file_info = bot.get_file(message.photo[-1].file_id) 
    else:
        print(doc_type(message.document))
        if doc_type(message.document) not in ['png', 'jpg', 'mp4']:#'gif']:
            bot.send_message(message.chat.id, "Только фото или видео!")
            return
        else:
            file_info = bot.get_file(message.document.file_id)
        
    downloaded_file = bot.download_file(file_info.file_path)
    
    # Saving file
    with open('data/{}.mp4'.format(message.chat.id), 'wb') as file:
        file.write(downloaded_file)
        file.close()
    
    labels = generate_labels(message.chat.id)
    

    with open("data/{}.txt".format(message.chat.id), 'w', encoding="utf-8") as file:
        file.write(str('\n'.join(labels)))
        file.close()
        
    bot.send_message(message.chat.id, '\n'.join(labels))
                     
    """TODO: return lyrics
    song = generate_song(labels)
    
    bot.send_message(message.chat.id, song)
    """
    


@bot.message_handler(commands=['new_song'])
def send_new_song(message):
    
    if not os.path.isfile("data/{}.txt".format(message.chat.id)):
        bot.send_message(message.chat.id, 'Я не могу петь без картинки :<')
        return
    
    with open("data/{}.txt".format(message.chat.id), 'r', encoding="utf-8") as file:
        labels = file.read()
        file.close()
        
    
    bot.send_message(message.chat.id, 'Пока только лейблы :P')
    bot.send_message(message.chat.id, labels)
    
    '''
    song = generate_song(labels.split())
    bot.send_message(message.chat.id, song)
    '''
    
    
    
@bot.message_handler(content_types=['text'])
def check(message):
    print(message.text)
    bot.send_message(message.chat.id, 'Пришлите фото или видео')

        


def startbot():
    bot.polling()
