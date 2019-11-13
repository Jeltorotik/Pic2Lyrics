# Pic2Lyrics

@VideoToSongBot 

Telegram-bot, generating poems in Russian language by given image/video in any format(png, jpg, mp4, gif, etc).
Used libraries:
1) Google API
2) Markovify
3) pytorch
4) cv2 
5) telebot

Idea:
The bot receives the image and with Google Vision recognizes all possible labels on the image.
Then, there are two possible ways to generate a poem:
1) By RNN language models that recieve labels as their input.
2) By Markov Chains trained on corpus of Russian Poems and song lyrics.

Examples:
1) ![example1](https://github.com/Jeltorotik/Pic2Lyrics/blob/master/examples/example1.jpg)
2) ![example2](https://github.com/Jeltorotik/Pic2Lyrics/blob/master/examples/example2.jpg)
3) ![example3](https://github.com/Jeltorotik/Pic2Lyrics/blob/master/examples/example3.jpg)

