import unittest

from prepare_data import remove_digits, remove_stop_words, lemmatize, tokenize


class PrepareTest(unittest.TestCase):
    rm_digits_text = 'это текст какого-то поста из тг https://hdjf.com/ghdjf-2345/ и это цифры 88 56 7гр7ншо87'
    
    def test_remove_digits(self):
        self.assertEqual(
            remove_digits(self.rm_digits_text),
            'это текст какого-то поста из тг https://hdjf.com/ghdjf-/ и это цифры   грншо'
        )

if __name__ == '__main__':
    unittest.main()