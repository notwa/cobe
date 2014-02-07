import unittest

from cobe.tokenizers import CobeStemmer, CobeTokenizer, MegaHALTokenizer

class testMegaHALTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MegaHALTokenizer()

    def testSplitEmpty(self):
        self.assertEquals(len(self.tokenizer.split("")), 0)

    def testSplitSentence(self):
        words = self.tokenizer.split("hi.")
        self.assertEquals(words, ["HI", "."])

    def testSplitComma(self):
        words = self.tokenizer.split("hi, cobe")
        self.assertEquals(words, ["HI", ", ", "COBE", "."])

    def testSplitImplicitStop(self):
        words = self.tokenizer.split("hi")
        self.assertEquals(words, ["HI", "."])

    def testSplitUrl(self):
        words = self.tokenizer.split("http://www.google.com/")
        self.assertEquals(words, ["HTTP", "://", "WWW", ".", "GOOGLE", ".", "COM", "/."])

    def testSplitApostrophe(self):
        words = self.tokenizer.split("hal's brain")
        self.assertEquals(words, ["HAL'S", " ", "BRAIN", "."])

        words = self.tokenizer.split("',','")
        self.assertEquals(words, ["'", ",", "'", ",", "'", "."])

    def testSplitAlphaAndNumeric(self):
        words = self.tokenizer.split("hal9000, test blah 12312")
        self.assertEquals(words, ["HAL", "9000", ", ", "TEST", " ", "BLAH", " ", "12312", "."])

        words = self.tokenizer.split("hal9000's test")
        self.assertEquals(words, ["HAL", "9000", "'S", " ", "TEST", "."])

    def testCapitalize(self):
        words = self.tokenizer.split("this is a test")
        self.assertEquals("This is a test.", self.tokenizer.join(words))

        words = self.tokenizer.split("A.B. Hal test test. will test")
        self.assertEquals("A.b. Hal test test. Will test.",
                          self.tokenizer.join(words))

        words = self.tokenizer.split("2nd place test")
        self.assertEquals("2Nd place test.", self.tokenizer.join(words))

class testCobeTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CobeTokenizer()

    def testSplitEmpty(self):
        self.assertEquals(len(self.tokenizer.split("")), 0)

    def testSplitSentence(self):
        words = self.tokenizer.split("hi.")
        self.assertEquals(words, ["hi", "."])

    def testSplitComma(self):
        words = self.tokenizer.split("hi, cobe")
        self.assertEquals(words, ["hi", ",", " ", "cobe"])

    def testSplitDash(self):
        words = self.tokenizer.split("hi - cobe")
        self.assertEquals(words, ["hi", " ", "-", " ", "cobe"])

    def testSplitMultipleSpacesWithDash(self):
        words = self.tokenizer.split("hi  -  cobe")
        self.assertEquals(words, ["hi", " ", "-", " ", "cobe"])

    def testSplitLeadingDash(self):
        words = self.tokenizer.split("-foo")
        self.assertEquals(words, ["-foo"])

    def testSplitLeadingSpace(self):
        words = self.tokenizer.split(" foo")
        self.assertEquals(words, ["foo"])

        words = self.tokenizer.split("  foo")
        self.assertEquals(words, ["foo"])

    def testSplitTrailingSpace(self):
        words = self.tokenizer.split("foo ")
        self.assertEquals(words, ["foo"])

        words = self.tokenizer.split("foo  ")
        self.assertEquals(words, ["foo"])

    def testSplitSmiles(self):
        words = self.tokenizer.split(":)")
        self.assertEquals(words, [":)"])

        words = self.tokenizer.split(";)")
        self.assertEquals(words, [";)"])

        # not smiles
        words = self.tokenizer.split(":(")
        self.assertEquals(words, [":("])

        words = self.tokenizer.split(";(")
        self.assertEquals(words, [";("])

    def testSplitUrl(self):
        words = self.tokenizer.split("http://www.google.com/")
        self.assertEquals(words, ["http://www.google.com/"])

        words = self.tokenizer.split("https://www.google.com/")
        self.assertEquals(words, ["https://www.google.com/"])

        # odd protocols
        words = self.tokenizer.split("cobe://www.google.com/")
        self.assertEquals(words, ["cobe://www.google.com/"])

        words = self.tokenizer.split("cobe:www.google.com/")
        self.assertEquals(words, ["cobe:www.google.com/"])

        words = self.tokenizer.split(":foo")
        self.assertEquals(words, [":", "foo"])

    def testSplitMultipleSpaces(self):
        words = self.tokenizer.split("this is  a test")
        self.assertEquals(words, ["this", " ", "is", " ", "a", " ", "test"])

    def testSplitVerySadFrown(self):
        words = self.tokenizer.split("testing :    (")
        self.assertEquals(words, ["testing", " ", ":    ("])

        words = self.tokenizer.split("testing          :    (")
        self.assertEquals(words, ["testing", " ", ":    ("])

        words = self.tokenizer.split("testing          :    (  foo")
        self.assertEquals(words, ["testing", " ", ":    (", " ", "foo"])

    def testSplitHyphenatedWord(self):
        words = self.tokenizer.split("test-ing")
        self.assertEquals(words, ["test-ing"])

        words = self.tokenizer.split(":-)")
        self.assertEquals(words, [":-)"])

        words = self.tokenizer.split("test-ing :-) 1-2-3")
        self.assertEquals(words, ["test-ing", " ", ":-)", " ", "1-2-3"])

    def testSplitApostrophes(self):
        words = self.tokenizer.split("don't :'(")
        self.assertEquals(words, ["don't", " ", ":'("])

    def testJoin(self):
        self.assertEquals("foo bar baz",
                          self.tokenizer.join(["foo", " ", "bar", " ", "baz"]))


class testCobeStemmer(unittest.TestCase):
    def setUp(self):
        self.stemmer = CobeStemmer("english")

    def testStemmer(self):
        self.assertEquals("foo", self.stemmer.stem("foo"))
        self.assertEquals("jump", self.stemmer.stem("jumping"))
        self.assertEquals("run", self.stemmer.stem("running"))

    def testStemmerCase(self):
        self.assertEquals("foo", self.stemmer.stem("Foo"))
        self.assertEquals("foo", self.stemmer.stem("FOO"))

        self.assertEquals("foo", self.stemmer.stem("FOO'S"))
        self.assertEquals("foo", self.stemmer.stem("FOOING"))
        self.assertEquals("foo", self.stemmer.stem("Fooing"))

if __name__ == '__main__':
    unittest.main()
