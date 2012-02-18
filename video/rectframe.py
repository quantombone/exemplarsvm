#Here is some wxPython code which draws a rectangle on the screen and
#moves it around, the goals is to highlight an object detection on the
#desktop.. more work needs to still be done
import wx
import signal

#def quit_gracefully(*args):
#    print 'quitting loop'
    
class FancyFrame(wx.Frame):
    def __init__(self, width, height):
        wx.Frame.__init__(self, None,
                          style = wx.STAY_ON_TOP |
                          wx.FRAME_NO_TASKBAR |
                          wx.FRAME_SHAPED,
                          size=(width, height))
        self.amount = 255;
        self.delta = -3;
        #self.SetTransparent(180)
        self.InitializeToSize(width,height,1,1)
        # b = wx.EmptyBitmap(width, height)
        # dc = wx.MemoryDC()
        # dc.SelectObject(b)
        # dc.SetBackground(wx.Brush('black'))
        # dc.Clear()
        # dc.SetBrush(wx.TRANSPARENT_BRUSH)
        # dc.SetPen(wx.Pen('red', 4))
        # dc.DrawRectangle(10, 10, width-20, height-20)
        # dc.SelectObject(wx.NullBitmap)
        # b.SetMaskColour('black')
        # self.SetShape(wx.RegionFromBitmap(b))

        
        # self.SetBackgroundColour('red')
        # self.Show(True)

        
        self.Bind(wx.EVT_KEY_UP, self.OnKeyDown)
        self.timer = wx.Timer(self)
        self.timer.Start(25)
        self.Bind(wx.EVT_TIMER, self.NullCycle)

        signal.signal(signal.SIGINT, self.AlphaCycle)

        #signal.signal(signal.SIGINT, self.AlphaCycle)

        # try:
        #     print 'starting loop'
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     quit_gracefully()


    def InitializeToSize(self,width,height,xtl,ytl):
        self.Show(False)
        b = wx.EmptyBitmap(width, height)
        dc = wx.MemoryDC()
        dc.SelectObject(b)
        dc.SetBackground(wx.Brush('black'))
        #dc.Clear()
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen('red', 4))
        dc.DrawRectangle(1, 1, width-1, height-1)
        dc.SelectObject(wx.NullBitmap)
        b.SetMaskColour('black')
        self.SetShape(wx.RegionFromBitmap(b))
        self.SetBackgroundColour('red')
        self.Move((xtl,ytl))
        self.Show(True)
        
    def AlphaCycle(self, evt, other=0):
        #print 'alpha cycle running'
        f = open("/tmp/coords.txt")
        xtl, ytl, xbr, ybr = "".join(f.readlines()).strip().split(" ")
        xtl = int(xtl)
        ytl = int(ytl)
        xbr = int(xbr)
        ybr = int(ybr)
        f.close()
        #self.amount += self.delta
        #if self.amount == 0 or self.amount == 255:
        #    self.delta = -self.delta
        
        if xbr-xtl == 1:
            #self.Hide()
            a=1
        else:
            self.InitializeToSize((xbr-xtl),ybr-ytl,xtl,ytl)
            self.Show()

    def NullCycle(self, evt, other=0):
        a = 10;
        #print 'null cycle running'


    def OnKeyDown(self, event):
        """quit if user press Esc"""
        if event.GetKeyCode() == 27:
            self.Close(force=True)
        else:
            event.Skip()

if __name__ == "__main__":


    app = wx.App()
    f = FancyFrame(1000, 1000)
    app.MainLoop()



