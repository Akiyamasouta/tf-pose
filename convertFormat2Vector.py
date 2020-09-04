import numpy as np

class CalculatAngle:
    def __init__(self):
        self.parts_id_pair = [[3, 4, 2], [2, 3, 1], [9, 10, 8]]
        self.MISSING_VALUE = None

    def convertFormat2Vector(self, humans):
        '''
        @param msg : ros-openpose format
        @return output : array(16dim, each joint angle)
        '''

        output = np.zeros((0, 3))
        for human in humans:
            angles = []
            
            '''
            confidence,
            part_id,
            x, y
            '''
            x = [0.0, 0.0, 0.0]
            y = [0.0, 0.0, 0.0]
            for p in self.parts_id_pair:
                #print(human)
                
                for i in range(3):
                    #print(p[i])
                    if p[i] not in human.body_parts.keys():
                        x[i] = 0.0
                        y[i] = 0.0


                    else:
                        x[i] = human.body_parts[p[i]].x
                        y[i] = human.body_parts[p[i]].y
                        
                '''
                else:
                    x0 = human.body_parts[p[0]].x
                    y0 = human.body_parts[p[0]].y
                    x1 = human.body_parts[p[1]].x
                    y1 = human.body_parts[p[1]].y
                    x2 = human.body_parts[p[2]].x
                    y2 = human.body_parts[p[2]].y
                '''
                
                #関節点がx0, y0
                if all([x[0], x[1], x[2], y[0], y[1], y[2]]):
                    u = np.array([x[1]-x[0], y[1]-y[0]])
                    v = np.array([x[2]-x[0], y[2]-y[0]])
                    angle = np.round(self.innerProduct(u, v))
                    angles.append(angle)
                else:
                    angles.append(self.MISSING_VALUE)
            
            #print(angles)
            output = np.vstack((angles, output))
            #print(output.shape)

        return output
        
    def innerProduct(self, u, v):
        '''
        calculation joint angle
        '''

        i = np.inner(u, v)
        n = np.linalg.norm(u) * np.linalg.norm(v)
        if n == 0:
            n = 0.001

        c = i / n
        deg = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        return deg