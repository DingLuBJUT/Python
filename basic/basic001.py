from itertools import chain,repeat,islice


def main():

    """
    迭代：遍历过程
    可迭代对象：实现__iter__、__getitem__方法的对象
    """

    class Library(object):
        def __init__(self):
            self.books = ['title', 'title2', 'title3']

        def __getitem__(self, i):
            # 使用下标遍历
            return self.books[i]

        def __len__(self):
            # 返回可迭代对象的长度
            return len(self.books)

        def __iter__(self):
            # 直接获取元素
            for titles in self.books:
                yield titles
    #  创建可迭代对象
    library = Library()
    # 直接获取元素迭代
    for e in library:
        print(e)
    # 使用下标遍历
    for i in range(len(library)):
        print(library[i])
    return



    """
    迭代器：实现 __iter__ 和 __next__ 方法的对象
    迭代器能在你调用next()方法的时候返回容器中的下一个值,迭代器就像一个懒加载的工厂,等到
    有人需要的时候才给它生成值返回没调用的时候就处于休眠状态等待下一次调用,直到无元素可调
    用并且返回StopIteration异常
    """

    class Library(object):
        def __init__(self):
            self.books = ['title', 'title2', 'title3']
            self.index = -1

        def __iter__(self):
            return self

        def __next__(self):
            self.index += 1
            if self.index >= len(self.books):
                raise StopIteration()
            return self.books[self.index]

    # 创建迭代器对象
    library = Library()
    # 通过next获取元素
    print(next(library))
    print(next(library))
    print(next(library))
    # 返回StopIteration异常,因为容器中已经没后元素
    print(next(library))

    # 直接遍历迭代器对象
    library = Library()
    for l in library:
        print(l)

    """
    通过iter()方法创建迭代器对象
    """
    data = iter([1,2,3,4])
    print(next(data))

    """
    chain、repeat、islice
        chain拉平列表
        repeat重复元素
        切分列表
    """
    # [1,2,3,4,5]
    list(chain(*[[1,2],[3],[4,5]])
    # repeat(10,5) [10,10,10,10,10]
    # islice(repeat(10,5),2) [10,10]
    islice(repeat(10,5),2)


    return


if __name__ == '__main__':
    main()
