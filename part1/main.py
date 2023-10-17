import pattern
import NumpyTests

'''
c = pattern.Checker(100, 25)
c.draw()
c.show()
NumpyTests.TestCheckers()

c = pattern.Circle(1024, 200, (512, 256))
c.draw()
c.show()
NumpyTests.TestCircle()

c = pattern.Spectrum(100)
c.draw()
c.show()
NumpyTests.TestSpectrum()
'''

from generator import ImageGenerator
gen = ImageGenerator('./exercise_data/','./Labels.json' , 15, [50, 50, 3], rotation=True, mirroring=True, shuffle=True)
#b1 = gen.next()[0]
gen.show()
gen.show()
gen.show()


