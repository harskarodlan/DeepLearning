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

