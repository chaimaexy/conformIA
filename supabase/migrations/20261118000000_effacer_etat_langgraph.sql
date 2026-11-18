-- ConformIA, phase 7c : supprimer une analyse efface aussi son état dans LangGraph.
--
-- L'état d'une analyse (profil, réponses, rapport) est dans le schéma `langgraph`, indexé par
-- l'identifiant de l'analyse (thread_id), sans clé étrangère : `on delete cascade` ne l'atteint
-- pas. Sans ce déclencheur, supprimer un utilisateur effaçait ses analyses mais laissait leur
-- état. L'API effaçait déjà l'état d'une analyse supprimée par DELETE /analyses/{id}.
--
-- La fonction est dans `langgraph`, schéma que l'API publique n'expose pas. Les tables de
-- LangGraph sont créées par le backend (PostgresSaver.setup()) : sans elles, rien à effacer.

create or replace function langgraph.effacer_etat_analyse()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
    if to_regclass('langgraph.checkpoints') is not null then
        delete from langgraph.checkpoint_writes where thread_id = old.id;
        delete from langgraph.checkpoint_blobs where thread_id = old.id;
        delete from langgraph.checkpoints where thread_id = old.id;
    end if;
    return old;
end;
$$;

revoke all on function langgraph.effacer_etat_analyse() from public, anon, authenticated;

create trigger analyses_effacer_etat_langgraph
    after delete on public.analyses
    for each row execute function langgraph.effacer_etat_analyse();
