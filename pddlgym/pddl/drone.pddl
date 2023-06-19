(define (domain drone)
    (:requirements :typing )
    (:types position direction)
    (:predicates 
        (drone-at ?v0 - position)
        (drone-to ?v0 - direction)
        (threat-at ?v0 - position)
        (movable-left ?v0 - position ?v1 - position ?v2 - direction ?v3 - direction)
        (movable-right ?v0 - position ?v1 - position ?v2 - direction ?v3 - direction)
        (movable-forward ?v0 - position ?v1 - position)
        ;(target-at ?v0 - position)
        ;(target-acquired)
    )
    (:constants 
        north - direction
        east - direction
        south - direction
        west - direction
    )
    (:action heading-forward
            :parameters (?from - position ?to - position ?dir - direction)
            :precondition (and (drone-at ?from )
                    (drone-to ?dir)
                    (not (threat-at ?to))
                    (movable-forward ?from ?to ?dir)
            )
            :effect (and
                    (not (drone-at ?from))
                    (drone-at ?to)
            )
    )
    (:action heading-left
            :parameters (?from - position ?to - position ?dir - direction ?newdir - direction)
            :precondition (and (drone-at ?from )
                    (drone-to ?dir)
                    (not (threat-at ?to))
                    (movable-left ?from ?to ?dir ?newdir)
            )
            :effect (and
                    (not (drone-at ?from))
                    (drone-at ?to)
                    (not (drone-to ?dir))
                    (drone-to ?newdir)
            )
    )
    (:action heading-right
            :parameters (?from - position ?to - position ?dir - direction ?newdir - direction)
            :precondition (and (drone-at ?from )
                    (drone-to ?dir)
                    (not (threat-at ?to))
                    (movable-right ?from ?to ?dir ?newdir) 
            )
            :effect (and
                    (not (drone-at ?from))
                    (drone-at ?to)
                    (not (drone-to ?dir))
                    (drone-to ?newdir)
            )
    )
)
