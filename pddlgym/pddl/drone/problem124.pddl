(define (problem drone)(:domain drone)
	(:objects
		pos-1-1 - position
		pos-1-2 - position
		pos-1-3 - position
		pos-1-4 - position
		pos-1-5 - position
		pos-2-1 - position
		pos-2-2 - position
		pos-2-3 - position
		pos-2-4 - position
		pos-2-5 - position
		pos-3-1 - position
		pos-3-2 - position
		pos-3-3 - position
		pos-3-4 - position
		pos-3-5 - position
		pos-4-1 - position
		pos-4-2 - position
		pos-4-3 - position
		pos-4-4 - position
		pos-4-5 - position
		pos-5-1 - position
		pos-5-2 - position
		pos-5-3 - position
		pos-5-4 - position
		pos-5-5 - position
	)
	(:init
		(drone-at pos-3-1)
		(drone-to south)
		(threat-at pos-4-3)
		(threat-at pos-2-3)
		(movable-forward pos-1-1 pos-1-2 north)
		(movable-right pos-1-1 pos-2-1 north east)
		(movable-forward pos-1-1 pos-2-1 east)
		(movable-left pos-1-1 pos-1-2 east north)
		(movable-left pos-1-1 pos-2-1 south east)
		(movable-right pos-1-1 pos-1-2 west north)
		(movable-forward pos-1-2 pos-1-3 north)
		(movable-right pos-1-2 pos-2-2 north east)
		(movable-forward pos-1-2 pos-2-2 east)
		(movable-right pos-1-2 pos-1-1 east south)
		(movable-left pos-1-2 pos-1-3 east north)
		(movable-forward pos-1-2 pos-1-1 south)
		(movable-left pos-1-2 pos-2-2 south east)
		(movable-right pos-1-2 pos-1-3 west north)
		(movable-left pos-1-2 pos-1-1 west south)
		(movable-forward pos-1-3 pos-1-4 north)
		(movable-right pos-1-3 pos-2-3 north east)
		(movable-forward pos-1-3 pos-2-3 east)
		(movable-right pos-1-3 pos-1-2 east south)
		(movable-left pos-1-3 pos-1-4 east north)
		(movable-forward pos-1-3 pos-1-2 south)
		(movable-left pos-1-3 pos-2-3 south east)
		(movable-right pos-1-3 pos-1-4 west north)
		(movable-left pos-1-3 pos-1-2 west south)
		(movable-forward pos-1-4 pos-1-5 north)
		(movable-right pos-1-4 pos-2-4 north east)
		(movable-forward pos-1-4 pos-2-4 east)
		(movable-right pos-1-4 pos-1-3 east south)
		(movable-left pos-1-4 pos-1-5 east north)
		(movable-forward pos-1-4 pos-1-3 south)
		(movable-left pos-1-4 pos-2-4 south east)
		(movable-right pos-1-4 pos-1-5 west north)
		(movable-left pos-1-4 pos-1-3 west south)
		(movable-right pos-1-5 pos-2-5 north east)
		(movable-forward pos-1-5 pos-2-5 east)
		(movable-right pos-1-5 pos-1-4 east south)
		(movable-forward pos-1-5 pos-1-4 south)
		(movable-left pos-1-5 pos-2-5 south east)
		(movable-left pos-1-5 pos-1-4 west south)
		(movable-forward pos-2-1 pos-2-2 north)
		(movable-right pos-2-1 pos-3-1 north east)
		(movable-left pos-2-1 pos-1-1 north west)
		(movable-forward pos-2-1 pos-3-1 east)
		(movable-left pos-2-1 pos-2-2 east north)
		(movable-right pos-2-1 pos-1-1 south west)
		(movable-left pos-2-1 pos-3-1 south east)
		(movable-forward pos-2-1 pos-1-1 west)
		(movable-right pos-2-1 pos-2-2 west north)
		(movable-forward pos-2-2 pos-2-3 north)
		(movable-right pos-2-2 pos-3-2 north east)
		(movable-left pos-2-2 pos-1-2 north west)
		(movable-forward pos-2-2 pos-3-2 east)
		(movable-right pos-2-2 pos-2-1 east south)
		(movable-left pos-2-2 pos-2-3 east north)
		(movable-forward pos-2-2 pos-2-1 south)
		(movable-right pos-2-2 pos-1-2 south west)
		(movable-left pos-2-2 pos-3-2 south east)
		(movable-forward pos-2-2 pos-1-2 west)
		(movable-right pos-2-2 pos-2-3 west north)
		(movable-left pos-2-2 pos-2-1 west south)
		(movable-forward pos-2-3 pos-2-4 north)
		(movable-right pos-2-3 pos-3-3 north east)
		(movable-left pos-2-3 pos-1-3 north west)
		(movable-forward pos-2-3 pos-3-3 east)
		(movable-right pos-2-3 pos-2-2 east south)
		(movable-left pos-2-3 pos-2-4 east north)
		(movable-forward pos-2-3 pos-2-2 south)
		(movable-right pos-2-3 pos-1-3 south west)
		(movable-left pos-2-3 pos-3-3 south east)
		(movable-forward pos-2-3 pos-1-3 west)
		(movable-right pos-2-3 pos-2-4 west north)
		(movable-left pos-2-3 pos-2-2 west south)
		(movable-forward pos-2-4 pos-2-5 north)
		(movable-right pos-2-4 pos-3-4 north east)
		(movable-left pos-2-4 pos-1-4 north west)
		(movable-forward pos-2-4 pos-3-4 east)
		(movable-right pos-2-4 pos-2-3 east south)
		(movable-left pos-2-4 pos-2-5 east north)
		(movable-forward pos-2-4 pos-2-3 south)
		(movable-right pos-2-4 pos-1-4 south west)
		(movable-left pos-2-4 pos-3-4 south east)
		(movable-forward pos-2-4 pos-1-4 west)
		(movable-right pos-2-4 pos-2-5 west north)
		(movable-left pos-2-4 pos-2-3 west south)
		(movable-right pos-2-5 pos-3-5 north east)
		(movable-left pos-2-5 pos-1-5 north west)
		(movable-forward pos-2-5 pos-3-5 east)
		(movable-right pos-2-5 pos-2-4 east south)
		(movable-forward pos-2-5 pos-2-4 south)
		(movable-right pos-2-5 pos-1-5 south west)
		(movable-left pos-2-5 pos-3-5 south east)
		(movable-forward pos-2-5 pos-1-5 west)
		(movable-left pos-2-5 pos-2-4 west south)
		(movable-forward pos-3-1 pos-3-2 north)
		(movable-right pos-3-1 pos-4-1 north east)
		(movable-left pos-3-1 pos-2-1 north west)
		(movable-forward pos-3-1 pos-4-1 east)
		(movable-left pos-3-1 pos-3-2 east north)
		(movable-right pos-3-1 pos-2-1 south west)
		(movable-left pos-3-1 pos-4-1 south east)
		(movable-forward pos-3-1 pos-2-1 west)
		(movable-right pos-3-1 pos-3-2 west north)
		(movable-forward pos-3-2 pos-3-3 north)
		(movable-right pos-3-2 pos-4-2 north east)
		(movable-left pos-3-2 pos-2-2 north west)
		(movable-forward pos-3-2 pos-4-2 east)
		(movable-right pos-3-2 pos-3-1 east south)
		(movable-left pos-3-2 pos-3-3 east north)
		(movable-forward pos-3-2 pos-3-1 south)
		(movable-right pos-3-2 pos-2-2 south west)
		(movable-left pos-3-2 pos-4-2 south east)
		(movable-forward pos-3-2 pos-2-2 west)
		(movable-right pos-3-2 pos-3-3 west north)
		(movable-left pos-3-2 pos-3-1 west south)
		(movable-forward pos-3-3 pos-3-4 north)
		(movable-right pos-3-3 pos-4-3 north east)
		(movable-left pos-3-3 pos-2-3 north west)
		(movable-forward pos-3-3 pos-4-3 east)
		(movable-right pos-3-3 pos-3-2 east south)
		(movable-left pos-3-3 pos-3-4 east north)
		(movable-forward pos-3-3 pos-3-2 south)
		(movable-right pos-3-3 pos-2-3 south west)
		(movable-left pos-3-3 pos-4-3 south east)
		(movable-forward pos-3-3 pos-2-3 west)
		(movable-right pos-3-3 pos-3-4 west north)
		(movable-left pos-3-3 pos-3-2 west south)
		(movable-forward pos-3-4 pos-3-5 north)
		(movable-right pos-3-4 pos-4-4 north east)
		(movable-left pos-3-4 pos-2-4 north west)
		(movable-forward pos-3-4 pos-4-4 east)
		(movable-right pos-3-4 pos-3-3 east south)
		(movable-left pos-3-4 pos-3-5 east north)
		(movable-forward pos-3-4 pos-3-3 south)
		(movable-right pos-3-4 pos-2-4 south west)
		(movable-left pos-3-4 pos-4-4 south east)
		(movable-forward pos-3-4 pos-2-4 west)
		(movable-right pos-3-4 pos-3-5 west north)
		(movable-left pos-3-4 pos-3-3 west south)
		(movable-right pos-3-5 pos-4-5 north east)
		(movable-left pos-3-5 pos-2-5 north west)
		(movable-forward pos-3-5 pos-4-5 east)
		(movable-right pos-3-5 pos-3-4 east south)
		(movable-forward pos-3-5 pos-3-4 south)
		(movable-right pos-3-5 pos-2-5 south west)
		(movable-left pos-3-5 pos-4-5 south east)
		(movable-forward pos-3-5 pos-2-5 west)
		(movable-left pos-3-5 pos-3-4 west south)
		(movable-forward pos-4-1 pos-4-2 north)
		(movable-right pos-4-1 pos-5-1 north east)
		(movable-left pos-4-1 pos-3-1 north west)
		(movable-forward pos-4-1 pos-5-1 east)
		(movable-left pos-4-1 pos-4-2 east north)
		(movable-right pos-4-1 pos-3-1 south west)
		(movable-left pos-4-1 pos-5-1 south east)
		(movable-forward pos-4-1 pos-3-1 west)
		(movable-right pos-4-1 pos-4-2 west north)
		(movable-forward pos-4-2 pos-4-3 north)
		(movable-right pos-4-2 pos-5-2 north east)
		(movable-left pos-4-2 pos-3-2 north west)
		(movable-forward pos-4-2 pos-5-2 east)
		(movable-right pos-4-2 pos-4-1 east south)
		(movable-left pos-4-2 pos-4-3 east north)
		(movable-forward pos-4-2 pos-4-1 south)
		(movable-right pos-4-2 pos-3-2 south west)
		(movable-left pos-4-2 pos-5-2 south east)
		(movable-forward pos-4-2 pos-3-2 west)
		(movable-right pos-4-2 pos-4-3 west north)
		(movable-left pos-4-2 pos-4-1 west south)
		(movable-forward pos-4-3 pos-4-4 north)
		(movable-right pos-4-3 pos-5-3 north east)
		(movable-left pos-4-3 pos-3-3 north west)
		(movable-forward pos-4-3 pos-5-3 east)
		(movable-right pos-4-3 pos-4-2 east south)
		(movable-left pos-4-3 pos-4-4 east north)
		(movable-forward pos-4-3 pos-4-2 south)
		(movable-right pos-4-3 pos-3-3 south west)
		(movable-left pos-4-3 pos-5-3 south east)
		(movable-forward pos-4-3 pos-3-3 west)
		(movable-right pos-4-3 pos-4-4 west north)
		(movable-left pos-4-3 pos-4-2 west south)
		(movable-forward pos-4-4 pos-4-5 north)
		(movable-right pos-4-4 pos-5-4 north east)
		(movable-left pos-4-4 pos-3-4 north west)
		(movable-forward pos-4-4 pos-5-4 east)
		(movable-right pos-4-4 pos-4-3 east south)
		(movable-left pos-4-4 pos-4-5 east north)
		(movable-forward pos-4-4 pos-4-3 south)
		(movable-right pos-4-4 pos-3-4 south west)
		(movable-left pos-4-4 pos-5-4 south east)
		(movable-forward pos-4-4 pos-3-4 west)
		(movable-right pos-4-4 pos-4-5 west north)
		(movable-left pos-4-4 pos-4-3 west south)
		(movable-right pos-4-5 pos-5-5 north east)
		(movable-left pos-4-5 pos-3-5 north west)
		(movable-forward pos-4-5 pos-5-5 east)
		(movable-right pos-4-5 pos-4-4 east south)
		(movable-forward pos-4-5 pos-4-4 south)
		(movable-right pos-4-5 pos-3-5 south west)
		(movable-left pos-4-5 pos-5-5 south east)
		(movable-forward pos-4-5 pos-3-5 west)
		(movable-left pos-4-5 pos-4-4 west south)
		(movable-forward pos-5-1 pos-5-2 north)
		(movable-left pos-5-1 pos-4-1 north west)
		(movable-left pos-5-1 pos-5-2 east north)
		(movable-right pos-5-1 pos-4-1 south west)
		(movable-forward pos-5-1 pos-4-1 west)
		(movable-right pos-5-1 pos-5-2 west north)
		(movable-forward pos-5-2 pos-5-3 north)
		(movable-left pos-5-2 pos-4-2 north west)
		(movable-right pos-5-2 pos-5-1 east south)
		(movable-left pos-5-2 pos-5-3 east north)
		(movable-forward pos-5-2 pos-5-1 south)
		(movable-right pos-5-2 pos-4-2 south west)
		(movable-forward pos-5-2 pos-4-2 west)
		(movable-right pos-5-2 pos-5-3 west north)
		(movable-left pos-5-2 pos-5-1 west south)
		(movable-forward pos-5-3 pos-5-4 north)
		(movable-left pos-5-3 pos-4-3 north west)
		(movable-right pos-5-3 pos-5-2 east south)
		(movable-left pos-5-3 pos-5-4 east north)
		(movable-forward pos-5-3 pos-5-2 south)
		(movable-right pos-5-3 pos-4-3 south west)
		(movable-forward pos-5-3 pos-4-3 west)
		(movable-right pos-5-3 pos-5-4 west north)
		(movable-left pos-5-3 pos-5-2 west south)
		(movable-forward pos-5-4 pos-5-5 north)
		(movable-left pos-5-4 pos-4-4 north west)
		(movable-right pos-5-4 pos-5-3 east south)
		(movable-left pos-5-4 pos-5-5 east north)
		(movable-forward pos-5-4 pos-5-3 south)
		(movable-right pos-5-4 pos-4-4 south west)
		(movable-forward pos-5-4 pos-4-4 west)
		(movable-right pos-5-4 pos-5-5 west north)
		(movable-left pos-5-4 pos-5-3 west south)
		(movable-left pos-5-5 pos-4-5 north west)
		(movable-right pos-5-5 pos-5-4 east south)
		(movable-forward pos-5-5 pos-5-4 south)
		(movable-right pos-5-5 pos-4-5 south west)
		(movable-forward pos-5-5 pos-4-5 west)
		(movable-left pos-5-5 pos-5-4 west south)
	)
	(:goal (drone-at pos-4-5))
)