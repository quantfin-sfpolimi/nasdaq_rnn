from helpermodules.df_cleaning import Timestamping
from datetime import datetime, timedelta

start = datetime(year=2022, month=11, day=29)
end = datetime(year=2022, month=11, day=30, hour= 23)
lista = [t for t in Timestamping(start, end)]
#print(lista[0], lista[-1])
print(len(lista))

generator = Timestamping(start_date=start, end_date=end)
boundaries = []
try:
    boundary_start = next(generator)
    boundary_end = boundary_start 
    while True: 
        for _ in range(9):
            try:
                boundary_end = next(generator)
            except StopIteration:
                boundaries.append((boundary_start, boundary_end))
                raise StopIteration
        boundaries.append((boundary_start, boundary_end))
        boundary_start = next(generator)   
except StopIteration:
    pass

for b in boundaries:
    print(b)
    

end = datetime.now()
start = end - timedelta(days=61)
print('Length for 2 months:', len([t for t in Timestamping(start, end)]))
 
 
