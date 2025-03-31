from Hinet.model import *
from Hinet.invblock import INV_block


class Hinet(nn.Module):

    def __init__(self):
        super(Hinet, self).__init__()

        self.inv1 = INV_block()
        self.inv2 = INV_block()
        self.inv3 = INV_block()
        self.inv4 = INV_block()
        # self.inv5 = INV_block()
        # self.inv6 = INV_block()
        # self.inv7 = INV_block()
        # self.inv8 = INV_block()
        # self.inv9 = INV_block()
        # self.inv10 = INV_block()
        # self.inv11 = INV_block()
        # self.inv12 = INV_block()
        # self.inv13 = INV_block()
        # self.inv14 = INV_block()
        # self.inv15 = INV_block()
        # self.inv16 = INV_block()

    def forward(self, x, rev=False):


        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            # out = self.inv5(out)
            # out = self.inv6(out)
            # out = self.inv7(out)
            # out = self.inv8(out)
            #
            # out = self.inv9(out)
            # out = self.inv10(out)
            # out = self.inv11(out)
            # out = self.inv12(out)
            # out = self.inv13(out)
            # out = self.inv14(out)
            # out = self.inv15(out)
            # out = self.inv16(out)

        else:
            # out = self.inv16(x, rev=True)
            # out = self.inv15(out, rev=True)
            # out = self.inv14(out, rev=True)
            # out = self.inv13(out, rev=True)
            # out = self.inv12(out, rev=True)
            # out = self.inv11(out, rev=True)
            # out = self.inv10(out, rev=True)
            # out = self.inv9(out, rev=True)
            #
            # out = self.inv8(x, rev=True)
            # out = self.inv7(out, rev=True)
            # out = self.inv6(x, rev=True)
            # out = self.inv5(out, rev=True)
            out = self.inv4(x, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)

        return out


