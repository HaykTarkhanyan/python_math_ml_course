from manim import *
import numpy as np



class Logo(Scene):
    def construct(self):
        
        self.camera.background_color = WHITE

        logo = SVGMobject("YSU_gerb-svg (4).svg").scale(1.7).shift(0.7*UP).set_color("#1e457c")
        ysu = Text('ԵՊՀ', font_size=35).shift(2.9*UP).set_color("#1e457c")
        mmf = VGroup(*[Text('ՄԱԹԵՄԱՏԻԿԱՅԻ', font_size=30).set_color("#1e457c"), Text('ԵՎ ՄԵԽԱՆԻԿԱՅԻ', font_size=30).set_color("#1e457c"), 
        Text('ՖԱԿՈՒԼՏԵՏ', font_size=30)]).set_color("#1e457c").arrange(direction=DOWN, buff=0.2).shift(2.3*DOWN)

        # self.add(logo, ysu, mmf)

        self.wait(0.3)
        self.play(Write(ysu, run_time=1))
        self.play(Write(logo.reverse_points(), run_time=4.5))
        self.wait()
        self.play(FadeOut(*[mob for mob in self.mobjects]), run_time=0.2)
        self.wait(0.2)