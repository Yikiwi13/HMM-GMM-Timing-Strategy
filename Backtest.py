import numpy as np

class backtest:
    """
    max_buy: maximum buy share
    max_sell: maximum sell share
    """
    max_buy = 1
    max_sell = 1

    def __init__(self, buy_fee=0, sell_fee=0, initial_money=10000, price=0, inventory=0):
        """
        initiator function
        :param buy_fee: transaction fee when buy, default 0
        :param sell_fee: transaction fee when sell, default 0
        :param initial_money: initial money position (in monetary units), default 100000
        :param price: current price, default 0
        :param inventory: current stock inventory (in shares), default 0
        """
        self.money = initial_money
        self.price = price
        self.inventory = inventory
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

    def buy(self,price,share=1):
        """
        buy function
        :param price: current price
        :param share: number of shares that want to buy, default 1
        :return: update the price, money, inventory , and value
        """
        self.price = price
        money_required = price * share * (1+self.buy_fee)
        if self.money < money_required:
            print("buy order exceed your current money position\n", "money position: ", self.money, "current price: ", self.price)
        else:
            if share>=self.max_buy:
                self.inventory += self.max_buy
                self.money -= price * self.max_buy * (1+self.buy_fee)
                print("buy {} stocks at {}\n".format(self.max_buy,price),"money position: ", self.money, "inventory: ",self.inventory, "total value: ", self.value)
            else:
                self.inventory += share
                self.money -= price * share * (1 + self.buy_fee)
                print("buy {} stocks at {}\n".format(share, price), "money position: ", self.money,
                      "inventory: ", self.inventory, "total value: ", self.value)


    def sell(self,price,share=1):
        """
        sell function
        :param price: current price
        :param share: number of shares that want to sell, default 1
        :return: update the price, money, inventory , and value
        """
        self.price = price
        if self.inventory < share:
            print("sell order exceed your current inventory\n", "inventory: ", self.inventory)
        else:
            if share>=self.max_sell:
                self.inventory -= self.max_sell
                self.money += price * self.max_sell * (1 - self.sell_fee)
                print("sell {} stocks at {}\n".format(self.max_sell, price), "money position: ", self.money,
                      "inventory: ", self.inventory, "total value: ", self.value)
            else:
                self.inventory -= share
                self.money += price * share * (1 - self.sell_fee)
                print("sell {} stocks at {}\n".format(share, price), "money position: ", self.money,
                      "inventory: ", self.inventory, "total value: ", self.value)

    def set_money(self,money):
        """
        change the money position
        """
        self.money = money

    @property
    def value(self):
        """
        defined as the sum of your current money position and stock position
        """
        return self.money + self.inventory * self.price

    @classmethod
    def set_max_buy(cls,max_buy):
        """
        reset the max_buy limit
        """
        cls.max_buy = max_buy

    @classmethod
    def set_max_sell(cls, max_sell):
        """
        reset the max_sell limit
        """
        cls.max_sell = max_sell

    @staticmethod
    def calc_maxDrawdown(curve):
        """
        input params
            curve: array or list, daily NAV sequence
        """
        i = np.argmax((np.maximum.accumulate(curve) - curve) / np.maximum.accumulate(curve))  # ending index
        if i == 0:
            return 0
        j = np.argmax(curve[:i])  # beginning index
        return (curve[j] - curve[i]) / (curve[j]),i,j

    @staticmethod
    def calc_sharpeRatio(ret, market_ret):
        """
        input param
            ret: array or list, daily return sequence
        """
        excess_ret = ret - market_ret
        mean_excess_ret = np.mean(excess_ret)
        sd_excess_ret = np.std(excess_ret)
        SR = np.sqrt(252) * mean_excess_ret / sd_excess_ret
        return SR


if __name__=='__main__':
    """
    S = backtest()
    S.set_money(9000)
    S.buy(5000,1)
    S.buy(5000,2)
    S.buy(10000,1)
    S.sell(5000,1)
    S.sell(5000,2)

    S2 = backtest(0.00005,0.00005) # transaction fee
    S2.set_money(9000)
    S2.buy(5000, 1)
    S2.buy(5000, 2)
    S2.buy(10000, 1)
    S2.sell(5000, 1)
    S2.sell(5000, 2)
    """



