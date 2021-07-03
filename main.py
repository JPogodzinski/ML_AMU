import numpy as np

def kwadraty(input_list):
    output_list = [pow(liczba,2) for liczba in input_list if liczba>0]
    return output_list

def wlasciwosci_macierzy(A):
    liczba_elementow =  A.size
    liczba_kolumn =  A.shape[1]
    liczba_wierszy =  A.shape[0]
    srednie_wg_wierszy =  A.mean(axis=1)
    srednie_wg_kolumn =  A.mean(axis=0)
    trzecia_kolumna =  A[:,2]
    czwarty_wiersz =  A[3]
    return (
        liczba_elementow, liczba_kolumn, liczba_wierszy,
        srednie_wg_wierszy, srednie_wg_kolumn,
        trzecia_kolumna, czwarty_wiersz)

def dzialanie1(A, x):
    result=np.dot(A,x)
    return result

def dzialanie2(A, B):
    result=np.dot(A,B)
    return result

def dzialanie3(A, B):
    result=np.linalg.det(np.dot(A,B))
    return result

def dzialanie4(A, B, x):
    result=np.dot(A,B).T-np.dot(B.T,A.T)
    return result



if __name__ == '__main__':
    input_list=[1,2,3,4,5, -2, 8]
    print(kwadraty(input_list))

    M = np.arange(1, 51).reshape(5,10)
    print(M)
    A=np.array([[0,4,-2],[-4,-3,0]])
    B=np.array([[0,1],[1,-1],[2,3]])
    x=np.array([2,1,0])


    print(wlasciwosci_macierzy(M))
    print(dzialanie1(A,x))
    print(dzialanie2(A,B))
    print(dzialanie3(A,B))
    print(dzialanie4(A,B,x))

    A = np.array([[1, 1], [1, 2]])
    print(A) #array A**-1 działa jak matematyczne podnoszenie liczb w tablicy do potęgi ujemnej  a nie odwracanie macierzy
    print(np.linalg.inv(A))
    matrix=np.matrix(A)
    print(matrix**-1)
    print("////////////////////////////////////////////////////")
    x=np.array([[1,2,3],[1,3,6]])
    y=np.array([5,6])
    result=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
    print(result)
    X=np.matrix(x)
    Y=np.matrix(y).T
    result=(X.T*X)**-1*X.T*Y
    print(result)