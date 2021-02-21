import pygame
from random import randint
import math
from sklearn.cluster import KMeans

def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
pygame.init()
pygame.display.set_caption("kmean visualization")

screen = pygame.display.set_mode((1200,700))
clock = pygame.time.Clock()
running = True

BACKGROUND = (124,124,124)
BLACK = (0,0,0)
BACKGROUND_PANEL = (249,255,230)
WHITE = (255,255,255)

RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (147, 153, 35)
PURPLE = (255,0,255)
SKY = (0,255,255)
ORANGE = (255,125,25)
GRAPE = (100,25,125)
GRASS = (55,155,65)

COLORS = [RED,GREEN,BLUE,YELLOW,PURPLE,SKY,ORANGE,GRAPE,GRASS]

font = pygame.font.SysFont('sans',40,True)
font_small = pygame.font.SysFont('sans',20,True)
text_plus = font.render('+',True,WHITE)
text_minus = font.render('-',True,WHITE)
text_run = font.render('Run',True,WHITE)
text_random = font.render('Random',True,WHITE)
text_error = font.render('Error',True,WHITE)
text_algorithm = font.render('Algorithm',True,WHITE)
text_reset = font.render('Reset',True,WHITE)

K=0
error=0
points = []
cluster = []
labels = []
while running:
    clock.tick(60)
    screen.fill(BACKGROUND)

    mouse_x, mouse_y = pygame.mouse.get_pos()
    # Draw interface
    # Draw panel
    pygame.draw.rect(screen,BLACK,(50,50,700,500))
    pygame.draw.rect(screen,BACKGROUND_PANEL,(55,55,690,490))

    # K Button +
    pygame.draw.rect(screen,BLACK,(850,50,50,50))
    screen.blit(text_plus,(860,55))

    # K button -
    pygame.draw.rect(screen, BLACK, (950, 50, 50, 50))
    screen.blit(text_minus, (960, 55))

    # K value
    text_K = font.render('K = ' + str(K), True, BLACK)
    screen.blit(text_K, (1050, 55))

    # Run button
    pygame.draw.rect(screen,BLACK,(850,150,150,50))
    screen.blit(text_run, (860, 155))

    # Random button
    pygame.draw.rect(screen, BLACK, (850, 250, 150, 50))
    screen.blit(text_random, (860, 255))



    # Algorithm button
    pygame.draw.rect(screen, BLACK, (850, 450, 150, 50))
    screen.blit(text_algorithm, (860, 455))

    # Reset button
    pygame.draw.rect(screen, BLACK, (850, 550, 150, 50))
    screen.blit(text_reset, (860, 555))

    # draw mouse position when mouse is in panel
    if 50 < mouse_x < 750 and 50 < mouse_y < 550:
        text_point = font_small.render('('+str(mouse_x-50)+','+str(mouse_y-50)+')', True, BLACK)
        screen.blit(text_point, (mouse_x+5, mouse_y))
    # End draw interface


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            # mouse is in pannel
            if 50 < mouse_x < 750 and 50 < mouse_y < 550:
                labels = []
                point = [mouse_x-50,mouse_y-50]
                points.append(point)
            # button K +
            if 850 < mouse_x < 900 and 50 < mouse_y < 100:
                if K<9:
                    K+=1

            # button K -
            if 950 < mouse_x < 1000 and 50 < mouse_y < 100:
                if K>0:
                    K-=1

            # run button
            if 850 < mouse_x < 1000 and 150 < mouse_y < 200:
                # assign point to closest cluster
                labels = []
                if points == [] or cluster == []:
                    continue
                for p in points:
                    distances_to_cluster = []
                    for c in cluster:
                        dis = distance(p,c)
                        distances_to_cluster.append(dis)
                    min_dis= min(distances_to_cluster)
                    labels.append(distances_to_cluster.index(min_dis))

                # update cluster
                for i in range(K):
                    sum_x = 0
                    sum_y = 0
                    count = 0
                    for j in range(len(points)):
                        if labels[j] == i:
                            sum_x += points[j][0]
                            sum_y += points[j][1]
                            count += 1
                    if count != 0:
                        cluster[i] = [sum_x/count, sum_y/count]
                # print('run')

            # random button
            if 850 < mouse_x < 1000 and 250 < mouse_y < 300:
                cluster = []
                labels = []
                for i in range(K):
                    cluster_random = [randint(0,700),randint(0,500)]
                    cluster.append(cluster_random)

            # algorithm button
            if 850 < mouse_x < 1000 and 450 < mouse_y < 500:
                try:
                    model = KMeans(n_clusters=K).fit(points)
                    cluster = model.cluster_centers_
                    labels = model.predict(points)
                except:
                    print('error')
                # print('algorithm')

            # reset button
            if 850 < mouse_x < 1000 and 550 < mouse_y < 600:
                K = 0
                error = 0
                points = []
                cluster = []
                labels = []
                # print('reset')
    # Draw cluster
    for i in range(len(cluster)):
        pygame.draw.circle(screen, COLORS[i], (cluster[i][0] + 50, cluster[i][1] + 50), 6)

    # Draw points in panel
    for i in range(len(points)):
        pygame.draw.circle(screen,BLACK,(points[i][0]+50,points[i][1]+50),6)
        if labels == []:
            pygame.draw.circle(screen, WHITE, (points[i][0] + 50, points[i][1] + 50), 5)
        else:
            pygame.draw.circle(screen, COLORS[labels[i]], (points[i][0] + 50, points[i][1] + 50), 5)

    # error text
    if points != [] and labels != []:
        error = 0
        for i in range(len(points)):
            error += distance(points[i], cluster[labels[i]])
    text_error = font.render('Error = ' + str(int(error)), True, BLACK)
    screen.blit(text_error, (850, 350))
    pygame.display.flip()
pygame.quit()