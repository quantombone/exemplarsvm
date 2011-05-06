#Here is some wxPython code which draws a rectangle on the screen and
#moves it around, the goals is to highlight an object detection on the
#desktop.. more work needs to still be done
import wx
print wx.version()

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
        self.InitializeToSize(width,height)
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
        self.Bind(wx.EVT_TIMER, self.AlphaCycle)

    def InitializeToSize(self,width,height):
        b = wx.EmptyBitmap(width, height)
        dc = wx.MemoryDC()
        dc.SelectObject(b)
        dc.SetBackground(wx.Brush('black'))
        dc.Clear()
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen('red', 4))
        dc.DrawRectangle(10, 10, width-20, height-20)
        dc.SelectObject(wx.NullBitmap)
        b.SetMaskColour('black')
        self.SetShape(wx.RegionFromBitmap(b))
        self.SetBackgroundColour('red')
        self.Show(True)
        
    def AlphaCycle(self, evt):
        self.amount += self.delta
        if self.amount == 0 or self.amount == 255:
            self.delta = -self.delta
        self.Move((self.amount,250))
        #print self.amount
        self.InitializeToSize(self.amount+100,220)
        self.Show()



    def OnKeyDown(self, event):
        """quit if user press Esc"""
        if event.GetKeyCode() == 27:
            self.Close(force=True)
        else:
            event.Skip()


if __name__ == "__main__":
    app = wx.App()
    f = FancyFrame(300, 300)
    app.MainLoop()

