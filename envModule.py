import math
import pygame
import random
import numpy as np
import gym
from gym import error, spaces, utils

class CarEnv(gym.Env):
    def __init__(self):
        self.DISPLAY_WIDTH = 1240
        self.DISPLAY_HEIGHT = 720
        self.DW_HALF = self.DISPLAY_WIDTH / 2
        self.DH_HALF = self.DISPLAY_HEIGHT / 2
        self.D2R = (math.pi * 2) / 360
        self.DIRECTION_VECTOR_LOOKUP = list([[math.cos(self.D2R * degrees), math.sin(self.D2R * degrees)] for degrees in range(360)])          
        self.DIRECTION_VECTOR_LOOKUP[90][0] = 0
        self.DIRECTION_VECTOR_LOOKUP[180][1] = 0
        self.DIRECTION_VECTOR_LOOKUP[270][0] = 0
        self.circle_x = 400
        self.circle_y = 260
        self.circle_move = True
        self.v=180
        self.mx=self.circle_x
        self.my=self.circle_y
        self.v = int(pygame.math.Vector2().angle_to((self.mx - self.circle_x, self.my - self.circle_y)))
        self.reward=0
        self.circle_r=25
        self.NP_x=self.circle_x+20
        self.NP_y=self.circle_y
        self.NP_x=self.circle_x-20
        self.NP_y=self.circle_y
        self.done=False
        self.prev_x=self.circle_x
        self.prev_y=self.circle_y
        self.action_space=spaces.Discrete(3)
        low = np.array([20,20,20,20,20],dtype=np.float32)
        high= np.array([520,520,520,520,520],dtype=np.float32)
        self.observation_space= spaces.Box(low,high,dtype=np.float32)
        self.gate=0
        self.OuterLine=[(50,400),(100,300),(250,200),(400,150),(800,180),(980,200),(1070,300),(1101,400),(1100,600),(1000,650),(900,700),(600,690),(400,691),(202,710),(51,600)]
        self.InnerLine=[(200,400),(300,350),(400,301),(500,300),(800,310),(900,320),(950,350),(970,400),(960,520),(930,550),(900,580),(600,560),(400,550),(250,580),(180,540)]#(200,400),(300,350),(400,301),(500,300),(800,320),(900,300),(950,350),(970,400),(1020,520),(1000,550),(900,560),(600,500),(400,550),(250,600),(180,540)
        self.acceleration=0
    def crash(self):
            return(-1,True)
    def through(self,OuterLine,InnerLine,NP_x,NP_y,circle_x,circle_y,v):
        def distances(circle_x,circle_y,v,lineSeg2,NP_x,NP_y):#lineSeg2 as wall, lineSeg1 as sensors
            distanceFromMid=1200
            circle_x2=(circle_x+((math.cos((v*3.14)/180)*500)))
            circle_y2=(circle_y+((math.sin((v*3.14)/180)*500)))
            lineSeg1=[(circle_x,circle_y),(circle_x2,circle_y2)]
            mid=[(circle_x,circle_y)]
            lineSeg1=[(circle_x,circle_y),(circle_x2,circle_y2)]
            line1_m=(lineSeg1[0][1]-lineSeg1[1][1])/(lineSeg1[0][0]-lineSeg1[1][0])
            line2_m=(lineSeg2[0][1]-lineSeg2[1][1])/(lineSeg2[0][0]-lineSeg2[1][0])
            line1_c=lineSeg1[0][1]-(line1_m*lineSeg1[0][0])
            line2_c=lineSeg2[0][1]-(line2_m*lineSeg2[0][0])
            if line1_m !=line2_m:
                x=(line2_c-line1_c)/(line1_m-line2_m)
                y=(line1_m*x)+line1_c
                if (math.sqrt(((circle_x-x)*(circle_x-x))+((circle_y-y)*(circle_y-y)))<math.sqrt(((NP_x-x)*(NP_x-x))+((NP_y-y)*(NP_y-y)))) and ((y<=lineSeg2[0][1] and y>=lineSeg2[1][1]) or ((y>=lineSeg2[0][1] and y<=lineSeg2[1][1]) and (x<=lineSeg2[0][0] and x>=lineSeg2[1][0])) or (x>=lineSeg2[0][0] and x<=lineSeg2[1][0])):
                    distanceFromMid= math.sqrt(((x-mid[0][0])*(x-mid[0][0]))+((y-mid[0][1])*(y-mid[0][1])))
                    #pygame.draw.circle(DS,(255,0,0),(int(x),int(y)),10)
            elif line1_m==line2_m:
                if (((lineSeg1[0][0]>lineSeg2[0][0] and lineSeg1[0][0]<lineSeg2[1][0]) or (lineSeg1[0][0]<lineSeg2[0][0] and lineSeg1[0][0]>lineSeg2[1][0])) or((lineSeg1[1][0]>lineSeg2[0][0] and lineSeg1[1][0]<lineSeg2[1][0]) or (lineSeg1[1][0]<lineSeg2[0][0] and lineSeg1[1][0]>lineSeg2[1][0]))) and((((lineSeg1[0][1]>lineSeg2[0][1] and lineSeg1[0][1]<lineSeg2[1][1]) or (lineSeg1[0][1]<lineSeg2[0][0] and lineSeg1[0][1]>lineSeg2[1][1])) or((lineSeg1[1][1]>lineSeg2[0][1] and lineSeg1[1][1]<lineSeg2[1][1]) or (lineSeg1[1][1]<lineSeg2[0][0] and lineSeg1[1][1]>lineSeg2[1][1])))):
                    parallel=[(math.sqrt(((mid[0][0]-lineSeg2[1][0])*(mid[0][0]-lineSeg2[1][0]))+((mid[0][1]-lineSeg2[1][1])*(mid[0][1]-lineSeg2[1][1]))),(math.sqrt(((mid[0][0]-lineSeg2[0][0])*(mid[0][0]-lineSeg2[0][0]))+((mid[0][1]-lineSeg2[0][1])*(mid[0][1]-lineSeg2[0][1])))))]
                    if parallel[0][0]<parallel[0][1]:
                        distanceFromMid=parallel[0][0]
            return(distanceFromMid)
        LIST=[]
        for p in range(0,len(OuterLine)-1):
            buffer=distances(circle_x,circle_y,v,[(OuterLine[p]),(OuterLine[p+1])],NP_x,NP_y)
            if buffer!=1200:
                    LIST.append(buffer)
            if p==0:
                buffer=distances(circle_x,circle_y,v,[(OuterLine[0]),(OuterLine[14])],NP_x,NP_y)
                if buffer!=1200:
                    LIST.append(buffer)
        for p in range(0,len(InnerLine)-1):
            buffer=distances(circle_x,circle_y,v,[(InnerLine[p]),(InnerLine[p+1])],NP_x,NP_y)
            if buffer!=1200:
                    LIST.append(buffer)
            if p==0:
                buffer=distances(circle_x,circle_y,v,[(InnerLine[0]),(InnerLine[14])],NP_x,NP_y)
                if buffer!=1200:
                    LIST.append(buffer)
        LIST.sort()
        return(LIST[0])
    def gatedR(self):
        reward=0
        if self.gate==0 and self.circle_y<360 and self.circle_x>550 and self.prev_x<550:
            self.gate+=1
            reward=1
        if self.gate==1 and self.circle_y<360 and self.circle_x>800 and self.prev_x<800:
            self.gate+=1
            reward=1
        if self.gate==2 and self.circle_x>700 and self.circle_y>400 and self.prev_y<400:
            self.gate+=1
            reward=1
        if self.gate==3 and self.circle_x>700 and self.circle_y>600 and self.prev_y<600:
            self.gate+=1
            reward=1
        if self.gate==4 and self.circle_y>360 and self.circle_x<800 and self.prev_x>800:
            self.gate+=1
            reward=1
        if self.gate==5 and self.circle_y>360 and self.circle_x<550 and self.prev_x>550:
            self.gate+=1
            reward=1
        if self.gate==6 and self.circle_x<500 and self.circle_y<600 and self.prev_y>600:
            self.gate+=1
            reward=1
        if self.gate==7 and self.circle_x<500 and self.circle_y<300 and self.prev_y>300:
            self.gate+=1
            reward=1
        return(reward,False,self.gate%8)
    def step(self,action):
        if action==1:
            self.v-=1
        elif action==2:
            self.v+=1
        elif action==0:
            self.v=self.v
            self.circle_move=True
    
        if self.v<0:
            self.v=360-(self.v*-1)
        elif self.v<=360:
            self.v=self.v%360
        self.prev_x=self.circle_x
        self.prev_y=self.circle_y
        if self.circle_move==True:
            self.circle_x+= self.DIRECTION_VECTOR_LOOKUP[self.v][0]*(1+self.acceleration)
            self.circle_y+= self.DIRECTION_VECTOR_LOOKUP[self.v][1]*(1+self.acceleration)
            if self.acceleration>0.05:
                self.acceleration=0.05
            else:
                self.acceleration+=0.005
            self.NP_x=self.circle_x-((self.DIRECTION_VECTOR_LOOKUP[self.v][0])*4)
            self.NP_y=self.circle_y-((self.DIRECTION_VECTOR_LOOKUP[self.v][1])*4)
            self.NP_2x=self.circle_x+((self.DIRECTION_VECTOR_LOOKUP[self.v][0])*4)
            self.NP_2y=self.circle_y+((self.DIRECTION_VECTOR_LOOKUP[self.v][1])*4)
            self.circle_move=False
        else:
            self.circle_x+= self.DIRECTION_VECTOR_LOOKUP[self.v][0]*(self.acceleration)
            self.circle_y+= self.DIRECTION_VECTOR_LOOKUP[self.v][1]*(self.acceleration)
            if self.acceleration>0.001:
                self.acceleration-=0.001
            else:
                self.acceleration=0.001
            self.NP_x=self.circle_x-((self.DIRECTION_VECTOR_LOOKUP[self.v][0])*4)
            self.NP_y=self.circle_y-((self.DIRECTION_VECTOR_LOOKUP[self.v][1])*4)
            self.NP_2x=self.circle_x+((self.DIRECTION_VECTOR_LOOKUP[self.v][0])*4)
            self.NP_2y=self.circle_y+((self.DIRECTION_VECTOR_LOOKUP[self.v][1])*4)
        def limit(distance,lim):
            if distance>=lim:
                distance=lim
            return distance
        a=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,(self.v-85)),500)
        b=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v-45),500)
        c=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v),500)
        d=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v+45),500)
        e=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v+85),500)
        #print(a,b,c,d,e)
        if a<23 or b<23 or c<23 or d<23 or e<23:
            self.reward,self.done=self.crash()
        else:
            self.reward, self.done, self.gate=(self.gatedR())
        return(np.array([a,b,c,d,e]),int(self.reward),self.done)
    def reset(self):
        self.circle_x = 400
        self.circle_y = 260
        self.mx=self.circle_x
        self.my=self.circle_y
        self.circle_move = True
        self.v=180
        self.v = int(pygame.math.Vector2().angle_to((self.mx - self.circle_x, self.my - self.circle_y)))
        self.reward=0
        self.NP_x=self.circle_x+20
        self.NP_y=self.circle_y
        self.NP_2x=self.circle_x-20
        self.NP_2y=self.circle_y
        self.done=False
        self.prev_x=self.circle_x
        self.prev_y=self.circle_y
        self.acceleration=0
        def limit(distance,lim):
            if distance>=lim:
                distance=lim
            return distance
        a=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,(self.v-85)),500)
        b=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v-45),500)
        c=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v),500)
        d=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v+45),500)
        e=limit(self.through(self.OuterLine,self.InnerLine,self.NP_x,self.NP_y,self.circle_x,self.circle_y,self.v+85),500)
        observations=[a,b,c,d,e]
        #print(observations)
        self.gate=0
        return(np.array(observations))
    def render(self):
        pygame.init()
        DISPLAY_WIDTH = 1240
        DISPLAY_HEIGHT = 720
        DW_HALF = DISPLAY_WIDTH / 2
        DH_HALF = DISPLAY_HEIGHT / 2
        pi=3.14
        constant=20
        DS = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
        def draw_track(OuterLine,InnerLine):
            for i in range(0,len(OuterLine)-1):
                pygame.draw.line(DS,(255,255,255),OuterLine[i],OuterLine[i+1],5)
                pygame.draw.line(DS,(255,255,255),OuterLine[0],OuterLine[14],5)
            for i in range(0,len(InnerLine)-1):
                pygame.draw.line(DS,(255,255,255),InnerLine[i],InnerLine[i+1],5)
                pygame.draw.line(DS,(255,255,255),InnerLine[0],InnerLine[14],5)
        pygame.draw.circle(DS, (0, 0, 255), (int(self.circle_x), int(self.circle_y)), self.circle_r, 0)

        pi=3.14
        constant=20
        lights=7
        arbitrary=[((self.circle_x+((math.cos((self.v*pi)/180)*constant)))+((math.cos(((self.v+90)*pi)/180)*constant)),(self.circle_y+((math.sin((self.v*pi)/180)*constant)))+((math.sin(((self.v+90)*pi)/180)*constant))),((self.circle_x-((math.cos((self.v*pi)/180)*constant)))+((math.cos(((self.v+90)*pi)/180)*constant)),(self.circle_y-((math.sin((self.v*pi)/180)*constant)))+((math.sin(((self.v+90)*pi)/180)*constant))),((self.circle_x+((math.cos((self.v*pi)/180)*constant)))-((math.cos(((self.v+90)*pi)/180)*constant)),(self.circle_y+((math.sin((self.v*pi)/180)*constant)))-((math.sin(((self.v+90)*pi)/180)*constant))),((self.circle_x-((math.cos((self.v*pi)/180)*constant)))-((math.cos(((self.v+90)*pi)/180)*constant)),(self.circle_y-((math.sin((self.v*pi)/180)*constant)))-((math.sin(((self.v+90)*pi)/180)*constant)))]
        pygame.draw.line(DS,(255,100,255),(arbitrary[0]),(arbitrary[1]),8)
        pygame.draw.line(DS,(255,100,255),(arbitrary[3]),(arbitrary[2]),8)
        pygame.draw.line(DS,(255,100,255),(arbitrary[3]),(arbitrary[1]),8)
        pygame.draw.line(DS,(200,0,255),(arbitrary[0]),(arbitrary[2]),8)
        pygame.draw.circle(DS, (240, 240, 240), (int(arbitrary[0][0]), int(arbitrary[0][1])), lights, 0)
        pygame.draw.circle(DS, (255, 0,0), (int(arbitrary[1][0]), int(arbitrary[1][1])), lights-2, 0)
        pygame.draw.circle(DS, (240, 240, 240), (int(arbitrary[2][0]), int(arbitrary[2][1])), lights, 0)
        pygame.draw.circle(DS, (255, 0,0), (int(arbitrary[3][0]), int(arbitrary[3][1])), lights-2, 0)
        
        draw_track(self.OuterLine,self.InnerLine)
        pygame.display.update()
        DS.fill((0, 0, 0))
