import numpy as np

def LoadBookData(book_dir="./"):
    """
    Loads book text from text file.

    Args:
        book_dir: directory of book text file
    Returns:
        book_data: string of complete book text
        unique_chars: list of unique characters
        char_to_ind: dict mapping character -> index
        ind_to_char: dict mapping index -> character
    """

    book_fname = book_dir + 'goblet_book.txt'
    fid = open(book_fname, "r")
    book_data = fid.read()
    fid.close()

    unique_chars = list(set(book_data))

    K = len(unique_chars)

    char_to_ind = {}
    ind_to_char = {}

    for i in range(K):
        c = unique_chars[i]
        char_to_ind[c] = i
        ind_to_char[i] = c
    
    return book_data, unique_chars, char_to_ind, ind_to_char 



def OneHotToStr(X, ind_to_char):
    """
    Convert one-hot encoded char index sequence to string.

    Args:
        X: sequence as one-hot encoded char indices (K, n)
        ind_to_char: dict mapping index -> character
    Returns: 
        text: X converted into string
    """
    n = X.shape[1]

    text = ""

    for i in range(n):
        x = X[:, i]
        idx = np.argmax(x)
        c = ind_to_char[idx]
        text = text + c
    
    return text



def StrToOneHot(text, char_to_ind, K):
    """
    Convert string to one-hot encoded char index sequence.

    Args:
        text: string of character sequence
        char_to_ind: dict mapping character -> index
        K: num of unique characters
    Returns: 
        X: text as one-hot encoded char indices (K, n)
    """
    n =  len(text)

    X = np.zeros((K, n))

    for i in range(n):
        c = text[i]
        idx = char_to_ind[c]
        X[idx, i] = 1
    
    return X