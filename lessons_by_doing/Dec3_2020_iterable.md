https://stackoverflow.com/a/7542261/8063281

## How to make a proper iterator 
- (if you want to iterate over chunk of tensor according to some criterion)
- Would you read the TensorDataset? (If that turns out a overkill for sth... then) 
- make own iterator for the DatasetClass that wraps the tensor 

### 1. \_\_getitem\_\_(self, idx)
- define <code>\_\_len\_\_</code> for later comfort. 
- mind that proper <code>IndexError</code> is expected for <code>StopIteration</code>
- <code>\_\_getitem\_\_</code> approach is compatible with <code>python 2.x</code>. 
### 2. \_\_iter\_\_(self) and \_\_next\_\_(self)
- this is newer standard but looks getitem is easier for use. 
- when proper <code>IndexError</code> isn't defined, try this. 

```python
# generator
def uc_gen(text):
    for char in text.upper():
        yield char

# generator expression
def uc_genexp(text):
    return (char for char in text.upper())

# iterator protocol
class uc_iter():
    def __init__(self, text):
        self.text = text.upper()
        self.index = 0
    def __iter__(self):
        return self
    def __next__(self):
        try:
            result = self.text[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

# getitem method
class uc_getitem():
    def __init__(self, text):
        self.text = text.upper()
    def __getitem__(self, index):
        return self.text[index]
```
