import time
import re

import openai

import pandas as pd


openai.api_key = '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'

posts = pd.read_excel('./Model/Notebooks/Data/posts (1).xlsx')['text']


pattern = r'^(?:(?:Финансы|Технологии|Политика|Шоубиз|Fashion|Крипта|Путешествия/релокация|Образовательный контент|Развлечения|Общее)(?:, )){29}(?:Финансы|Технологии|Политика|Шоубиз|Fashion|Крипта|Путешествия/релокация|Образовательный контент|Развлечения|Общее)$'
N = 30
def main():
    skip_times = 0

    for i in range(0, len(posts), N):
        print(f'{skip_times=}')
        i = i - skip_times * N
        print(i)

        completion = openai.ChatCompletion.create(
            model='gpt-3.5-turbo-16k',
            messages=[
                {'role': 'user', 'content': f'''
Тебе будет передано {N} постов из каналов в социальной сети Телеграм, разделённые этим набором символов: `u&*$)G`, каждый из которых ты должен отнести к одной из 10 категорий:
Финансы, Технологии, Политика, Шоубиз, Fashion, Крипта, Путешествия/релокация, Образовательный контент, Развлечения, Общее.
Твоя задача: соотнести каждый пост с одной из переданых тебе категорий, основываясь на содержании поста и его тематике
Твоим ответом должна быть одна строка, без какого-либо лишнего текста, 
которая содержит в себе {N} и только {N} категорий, только лишь из того списка, 
который я тебе передал, разделённые одной запятой с пробелом. Посты: {'u&*$)G'.join(posts[i:i+N])}

'''}
            ]
        )
        with open('cats.txt', 'a', encoding='utf-8') as f:
            print(completion.choices[0].message.content)
            text = completion.choices[0].message.content

            if re.match(pattern, text):
                f.write(f', {text}')
            else:
                if len(text.split(', ')) == 29:
                    f.write(f', {text}, null')
                elif len(splited_text := text.split(', ')) == 31:
                    f.write(f', {", ".join(splited_text[:30])}')
                else:
                    print(f"категории: {len(text.split(', '))}")
                    skip_times += 1

        time.sleep(60)

if __name__ == '__main__':
    main()

'''
1) написать запрос для чата с списком категорий, которые необхоимо распределить этим 15 постам
2) сказать, что посты разделены наобром символов: '87^F8tG' и каждому 
из постов нужно присвоить категорию и ответить одной строкой без какого-либо 
лишнего текста с категориями разделёнными одним пробелом.
'''